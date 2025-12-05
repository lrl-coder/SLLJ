import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import matplotlib

matplotlib.use('Agg')  # 无头模式，防止服务器报错

from dataset import LocalFlickrDataset
from attacker import MultimodalAttacker
from utils import setup_seed, visualize_attack_result


# ============================
# 1. 全局评估函数 (R@K 计算)
# ============================
def evaluate_global_retrieval(image_embeds, text_embeds, k_list=[1, 3, 5, 10]):
    """
    计算全局检索指标 (R@K, Mean Rank)
    Args:
        image_embeds: 所有测试图像的特征 (N, D)
        text_embeds: 所有候选文本的特征 (N, D)
    """
    # 归一化
    image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=1)
    text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=1)

    # 计算相似度矩阵 (N_imgs, N_texts)
    # 也就是每一张图和库里所有文本算相似度
    sim_matrix = torch.matmul(image_embeds, text_embeds.t())

    num_samples = sim_matrix.shape[0]
    metrics = {}

    # 获取每个图像对应的 Ground Truth 文本的排名
    # 假设测试集中第 i 张图的正确描述就是第 i 个文本
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)

    # 构建 GT 索引: [[0], [1], [2], ...]
    gt_indices = torch.arange(num_samples, device=sim_matrix.device).view(-1, 1)

    # 找到 GT 在排序结果中的位置
    matches = (sorted_indices == gt_indices)
    ranks = matches.nonzero()[:, 1]  # 获取 rank (0-based)

    # 计算 R@K
    for k in k_list:
        recall = (ranks < k).float().mean().item()
        metrics[f"R@{k}"] = recall * 100

    # 计算 Mean Rank
    metrics["Mean Rank"] = (ranks.float() + 1).mean().item()

    return metrics


# ============================
# 2. 主函数
# ============================
def main():
    # --- A. 基础配置 ---
    setup_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 数据集路径 (请根据您的实际环境修改)
    dataset_config = {
        "dataset_root": r"../dataset/flickr30k_images",
        "ann_file": r"../dataset/flickr30k_images/results.csv",
        "max_samples": 16,  # 论文建议跑 1000 张测试图
        "batch_size": 16
    }

    # --- B. 定义对比实验配置 ---
    # 这就是您论文核心实验的配置列表
    configurations = [
        # 1. FGSM: 单步攻击，步长直接设为 epsilon，无动量
        {"name": "FGSM", "epsilon": 8 / 255, "alpha": 8 / 255, "steps": 1, "decay": 0.0},

        # 2. PGD: 多步迭代，步长较小，无动量
        {"name": "PGD", "epsilon": 8 / 255, "alpha": 2 / 255, "steps": 10, "decay": 0.0},

        # 3. MI-FGSM: 多步迭代，带 1.0 的动量 (我们的最强攻击)
        {"name": "MI-FGSM", "epsilon": 8 / 255, "alpha": 2 / 255, "steps": 10, "decay": 1.0}
    ]

    # --- C. 数据与模型加载 ---
    images_dir = os.path.join(dataset_config["dataset_root"], "flickr30k_images")
    dataset = LocalFlickrDataset(
        images_dir=images_dir,
        ann_file=dataset_config["ann_file"],
        max_samples=dataset_config["max_samples"],
        delimiter='|'
    )

    if len(dataset) == 0:
        print("Error: 数据集加载为空，请检查路径。")
        return

    dataloader = DataLoader(dataset, batch_size=dataset_config["batch_size"], shuffle=False, num_workers=0)

    print("\n--- Loading CLIP Model ---")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    attacker = MultimodalAttacker(model, processor, device)

    # 结果存储列表
    final_results = []

    # --- D. 首先评估 Clean (无攻击) 性能 ---
    print("\n[Evaluating Clean Performance...]")
    clean_img_feats = []
    text_feats = []

    # 只需要跑一次 Clean 特征提取
    for images, texts in dataloader:
        images = images.to(device)
        with torch.no_grad():
            # 提取 Image 特征
            norm_imgs = attacker.normalizer(images)
            img_emb = model.get_image_features(pixel_values=norm_imgs)
            clean_img_feats.append(img_emb.cpu())

            # 提取 Text 特征
            txt_emb = attacker.get_text_features(list(texts))
            text_feats.append(txt_emb.cpu())

    clean_img_feats = torch.cat(clean_img_feats, dim=0).to(device)
    text_feats = torch.cat(text_feats, dim=0).to(device)

    clean_metrics = evaluate_global_retrieval(clean_img_feats, text_feats)
    print(f"Clean Metrics: R@1={clean_metrics['R@1']:.2f}%, MR={clean_metrics['Mean Rank']:.2f}")

    # 把 Clean 结果也加入表格
    final_results.append({
        "Method": "Clean (No Attack)",
        "R@1": clean_metrics['R@1'],
        "R@3": clean_metrics['R@3'],
        "R@5": clean_metrics['R@5'],
        "R@10": clean_metrics['R@10'],
        "Mean Rank": clean_metrics['Mean Rank']
    })

    # --- E. 循环运行三种攻击 ---
    print("\n--- Starting Comparative Experiment ---")

    for config in configurations:
        print(f"\n>>> Running Attack: {config['name']} ...")

        adv_img_feats = []

        # 遍历数据集生成对抗样本
        for i, (images, texts) in enumerate(dataloader):
            images = images.to(device)

            # 获取目标特征：这里我们做 Untargeted Attack (无定向攻击)
            # 目标是让图像远离它原本的 Ground Truth 文本
            # 所以 target_text_embeds 就是它原本的文本特征
            with torch.no_grad():
                current_text_feats = attacker.get_text_features(list(texts))

            # 生成对抗样本
            adv_images = attacker.mi_fgsm_attack(
                images,
                current_text_feats,  # 传入真实文本
                epsilon=config["epsilon"],
                alpha=config["alpha"],
                steps=config["steps"],
                decay=config["decay"],
                targeted=False  # False 表示我们要 maximize distance
            )

            # 提取对抗样本特征
            with torch.no_grad():
                norm_adv = attacker.normalizer(adv_images)
                adv_emb = model.get_image_features(pixel_values=norm_adv)
                adv_img_feats.append(adv_emb.cpu())

            # 保存第一张图看看效果
            if i == 0:
                save_name = f"vis_{config['name']}.png"
                visualize_attack_result(images[0], adv_images[0], save_name=save_name)

        # 拼接特征并评估
        adv_img_feats = torch.cat(adv_img_feats, dim=0).to(device)
        adv_metrics = evaluate_global_retrieval(adv_img_feats, text_feats)

        print(f"[{config['name']}] Result: R@1={adv_metrics['R@1']:.2f}%, MR={adv_metrics['Mean Rank']:.2f}")

        # 记录结果
        res_dict = {"Method": config['name']}
        res_dict.update(adv_metrics)
        final_results.append(res_dict)

    # --- F. 打印最终论文表格 ---
    print("\n" + "=" * 50)
    print("FINAL EXPERIMENTAL RESULTS (Copy to your paper)")
    print("=" * 50)

    df = pd.DataFrame(final_results)
    # 调整列顺序
    cols = ["Method", "R@1", "R@3", "R@5", "R@10", "Mean Rank"]
    print(df[cols].to_markdown(index=False, floatfmt=".2f"))
    print("=" * 50)


if __name__ == "__main__":
    main()