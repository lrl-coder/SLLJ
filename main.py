import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import matplotlib

# 如果在服务器无图形界面环境运行，取消下一行的注释
# matplotlib.use('Agg')
matplotlib.use('TkAgg')  # Windows 本地运行推荐 TkAgg

# 导入自定义模块
from dataset import LocalFlickrDataset, MiniCOCODataset, DifferentiableNormalize, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from attacker import MultimodalAttacker
from utils import setup_seed, visualize_attack_result


# ============================
# 1. 增强评估函数 (保持不变)
# ============================
def evaluate_attack_performance(model, images, true_texts, target_texts, device, processor, prefix=""):
    """
    更详细的评估：计算 ASR (攻击成功率) 以及 相似度变化指标
    """
    normalizer = DifferentiableNormalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD).to(device)

    with torch.no_grad():
        # 1. 获取图像特征
        norm_imgs = normalizer(images)
        img_embeds = model.get_image_features(pixel_values=norm_imgs)
        img_embeds = torch.nn.functional.normalize(img_embeds, p=2, dim=1)

        # 2. 获取文本特征 (True Text & Target Text)
        true_inputs = processor(text=true_texts, return_tensors="pt", padding=True).to(device)
        target_inputs = processor(text=target_texts, return_tensors="pt", padding=True).to(device)

        true_embeds = model.get_text_features(**true_inputs)
        target_embeds = model.get_text_features(**target_inputs)

        true_embeds = torch.nn.functional.normalize(true_embeds, p=2, dim=1)
        target_embeds = torch.nn.functional.normalize(target_embeds, p=2, dim=1)

        # 3. 计算相似度
        sim_to_true = (img_embeds * true_embeds).sum(dim=1)  # [Batch]
        sim_to_target = (img_embeds * target_embeds).sum(dim=1)  # [Batch]

        # 4. 计算 ASR
        success_mask = sim_to_target > sim_to_true
        asr = success_mask.float().mean().item()

        # 5. 统计平均相似度
        avg_sim_true = sim_to_true.mean().item()
        avg_sim_target = sim_to_target.mean().item()

        print(f"[{prefix}] ASR: {asr * 100:.2f}% | "
              f"Avg Sim to True: {avg_sim_true:.4f} | "
              f"Avg Sim to Target: {avg_sim_target:.4f}")

        return asr, avg_sim_true, avg_sim_target


# ============================
# 2. 主函数
# ============================
def main():
    # A. 基础设置
    setup_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 攻击超参数配置
    attack_config = {
        "epsilon": 8 / 255,  # 最大扰动限制
        "alpha": 2 / 255,  # 单步步长
        "steps": 20,  # 迭代次数
        "decay": 1.0,  # 动量因子
        "targeted": True  # 是否为定向攻击
    }

    # B. 准备数据
    print("\n--- Initializing Data from Local Disk ---")

    # === 请修改这里的路径为你自己的真实路径 ===
    dataset_root = r"../dataset/flickr30k_images"
    images_dir = os.path.join(dataset_root, "flickr30k_images")
    ann_file = os.path.join(dataset_root, "results.csv")

    # 实例化数据集
    dataset = LocalFlickrDataset(
        images_dir=images_dir,
        ann_file=ann_file,
        max_samples=1000,  # 调试模式只跑1000张
        delimiter='|'  # Flickr30k 标准分隔符
    )

    if len(dataset) > 0:
        # Batch Size = 2，意味着每次处理2张图
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    else:
        print("Error: 数据集为空，请检查路径是否正确。")
        return

    # C. 加载模型
    print("\n--- Loading CLIP Model ---")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    attacker = MultimodalAttacker(model, processor, device)

    # D. 实验循环
    print("\n--- Starting Attack Experiment ---")
    print(f"Config: {attack_config}")

    for i, (images, texts) in enumerate(dataloader):
        images = images.to(device)
        current_batch_size = len(images)  # 动态获取当前 batch 大小 (防止最后一批不足2张)

        # =======================================================
        # 核心优化：动态生成 Target Captions
        # =======================================================

        # 策略 1: 错位攻击 (Permutation Attack) [学术推荐]
        # 目标是：第1张图去匹配第2张图的文本，第2张去匹配第1张...
        # 这样能证明模型被彻底混淆了，而且不需要编造文本。
        target_captions = list(texts[1:]) + [texts[0]]

        # 策略 2: 固定目标攻击 (Fixed Target Attack) [可选]
        # 如果你想看“把所有东西都变成蜘蛛侠”，取消下面两行的注释
        # target_concept = "a photo of spiderman"
        # target_captions = [target_concept] * current_batch_size

        # =======================================================

        print(f"\nBatch {i} Targets Example: '{texts[0]}' -> '{target_captions[0]}'")

        # 预计算目标文本特征 (注意：现在 target_captions 的长度严格等于 current_batch_size)
        target_emb = attacker.get_text_features(target_captions)

        # --- 1. 攻击前评估 (Baseline) ---
        # 注意：这里传入的是 batch 专属的 target_captions
        evaluate_attack_performance(
            model, images, list(texts), target_captions, device, processor, prefix="Clean"
        )

        # --- 2. 执行 MI-FGSM 攻击 ---
        adv_images = attacker.mi_fgsm_attack(
            images,
            target_emb,
            epsilon=attack_config["epsilon"],
            alpha=attack_config["alpha"],
            steps=attack_config["steps"],
            decay=attack_config["decay"],
            targeted=attack_config["targeted"]
        )

        # --- 3. 验证扰动约束 ---
        with torch.no_grad():
            diff = (adv_images - images).abs()
            max_diff = diff.view(current_batch_size, -1).max(dim=1)[0].mean().item()
            print(f"[Sanity Check] Max perturbation (L_inf): {max_diff:.4f} (Limit: {attack_config['epsilon']:.4f})")

        # --- 4. 攻击后评估 ---
        evaluate_attack_performance(
            model, adv_images, list(texts), target_captions, device, processor, prefix="Adv"
        )

        # --- 5. 可视化 ---
        save_path = f"result_batch_{i}.png"
        visualize_attack_result(images[0], adv_images[0], save_name=save_path)


if __name__ == "__main__":
    main()