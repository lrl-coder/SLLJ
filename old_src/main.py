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
from utils import setup_seed, visualize_attack_result, evaluate_global_retrieval , plot_performance_comparison
# 新增：导入防御函数
from defence import jpeg_compress_defense




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
        # 论文建议跑 1000 张测试图，调试时可设小一点，如 16 或 100
        "max_samples": 100,
        "batch_size": 16
    }

    # --- B. 定义对比实验配置 ---
    # 这就是您论文核心实验的配置列表
    configurations = [
        # 1. FGSM: 单步攻击
        {"name": "FGSM", "epsilon": 8 / 255, "alpha": 8 / 255, "steps": 1, "decay": 0.0},

        # 2. PGD: 多步迭代
        {"name": "PGD", "epsilon": 8 / 255, "alpha": 2 / 255, "steps": 10, "decay": 0.0},

        # 3. MI-FGSM: 多步迭代 + 动量
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

    # --- E. 循环运行三种攻击 + 防御 ---
    print("\n--- Starting Comparative Experiment (Attack & Defense) ---")

    for config in configurations:
        print(f"\n>>> Running Attack: {config['name']} ...")

        adv_img_feats = []
        defended_img_feats = []  # 新增：存储防御后的图像特征

        # 遍历数据集生成对抗样本
        for i, (images, texts) in enumerate(dataloader):
            images = images.to(device)

            # 获取目标特征：无定向攻击
            with torch.no_grad():
                current_text_feats = attacker.get_text_features(list(texts))

            # 1. 生成对抗样本
            adv_images = attacker.mi_fgsm_attack(
                images,
                current_text_feats,
                epsilon=config["epsilon"],
                alpha=config["alpha"],
                steps=config["steps"],
                decay=config["decay"],
                targeted=False
            )

            # 2. 实施防御：JPEG 压缩
            # 由于 defence.py 里的函数处理单张 Tensor，这里需要循环处理 Batch
            defended_images_batch = []
            for j in range(adv_images.size(0)):
                # quality=50 是比较强的压缩，能更好体现防御效果
                def_img = jpeg_compress_defense(adv_images[j], quality=50)
                defended_images_batch.append(def_img)
            defended_images = torch.stack(defended_images_batch)

            # 3. 提取特征
            with torch.no_grad():
                # 3.1 提取对抗样本特征 (Attack)
                norm_adv = attacker.normalizer(adv_images)
                adv_emb = model.get_image_features(pixel_values=norm_adv)
                adv_img_feats.append(adv_emb.cpu())

                # 3.2 提取防御后样本特征 (Defense)
                norm_defended = attacker.normalizer(defended_images)
                def_emb = model.get_image_features(pixel_values=norm_defended)
                defended_img_feats.append(def_emb.cpu())

            # 保存第一张图看看效果
            if i == 0:
                save_name = f"vis_{config['name']}.png"
                visualize_attack_result(images[0], adv_images[0], save_name=save_name)

        # --- 评估并记录结果 ---

        # 1. 评估纯攻击效果
        adv_img_feats = torch.cat(adv_img_feats, dim=0).to(device)
        adv_metrics = evaluate_global_retrieval(adv_img_feats, text_feats)
        print(f"[{config['name']}] Attack Result: R@1={adv_metrics['R@1']:.2f}%")

        res_dict = {"Method": config['name']}
        res_dict.update(adv_metrics)
        final_results.append(res_dict)

        # 2. 评估防御后效果 (JPEG)
        defended_img_feats = torch.cat(defended_img_feats, dim=0).to(device)
        def_metrics = evaluate_global_retrieval(defended_img_feats, text_feats)
        print(f"[{config['name']} + JPEG] Defense Result: R@1={def_metrics['R@1']:.2f}%")

        res_dict_def = {"Method": config['name'] + " + JPEG"}
        res_dict_def.update(def_metrics)
        final_results.append(res_dict_def)

    # --- F. 打印最终论文表格 ---
    print("\n" + "=" * 60)
    print("FINAL EXPERIMENTAL RESULTS (Copy to your paper)")
    print("=" * 60)

    df = pd.DataFrame(final_results)
    # 调整列顺序
    cols = ["Method", "R@1", "R@3", "R@5", "R@10", "Mean Rank"]
    print(df[cols].to_markdown(index=False, floatfmt=".2f"))
    print("=" * 60)

    # 自动绘图调用
    plot_performance_comparison(final_results, save_name="experiment_summary.png")
    print("\n[Success] Experiment summary chart saved as 'experiment_summary.png'")


if __name__ == "__main__":
    main()