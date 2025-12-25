import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import torch
import numpy as np
import pandas as pd
import json
import datetime
import shutil
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import matplotlib
from tqdm import tqdm
from collections import defaultdict
from datetime import timezone, timedelta
matplotlib.use('Agg')

from utils.dataset import LocalFlickrDataset
from utils.attacker import MultimodalAttacker
from utils.utils import setup_seed, visualize_attack_result, evaluate_global_retrieval, plot_performance_comparison
from utils.defence import jpeg_compress_defense, batch_jpeg_compress_defense
from config.eval_dataset_config import dataset_config
from config.algorithm_config import configurations

torch.backends.cudnn.benchmark = True

model_list = ['clip-vit-base-patch32', 'clip-vit-large-patch14', 'clip-vit-base-patch16', 'clip-vit-large-patch14-336']

def get_top1_text_prediction(image_embed, text_embeds, all_texts):
    """
    辅助函数：给定一张图片的特征，在所有文本中检索相似度最高的文本
    """
    # image_embed: (1, D)
    # text_embeds: (N_texts, D)
    # Normalize
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Cosine Similarity
    sim = torch.matmul(image_embed, text_embeds.t()) # (1, N_texts)
    
    # Get Top-1 Index
    top1_idx = torch.argmax(sim).item()
    return all_texts[top1_idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="clip-vit-large-patch14", help="JPEG quality for defense (default: 50)")
    parser.add_argument("--jpeg_quality", type=int, default=50, help="JPEG quality for defense (default: 50)")
    args = parser.parse_args()
  
    # 1. 实验初始化与目录创建
    setup_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    # 创建带时间戳的实验目录
    cn_timezone = timezone(timedelta(hours=8))
    timestamp = datetime.datetime.now(cn_timezone).strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join("results", f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"--- Experiment Results will be saved to: {exp_dir} ---")

    model_id = f'openai/{args.model}'
    jpeg_quality = args.jpeg_quality
    # --- 根据模型 ID 自动确定分辨率 ---
    if "clip-vit-large-patch14-336" in model_id:
        img_size = 336
    else:
        img_size = 224

    # 2. 保存配置文件 (JSON)
    config_save_path = os.path.join(exp_dir, "config.json")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model_id": model_id,
            "dataset_config": dataset_config,
            "attack_configurations": configurations,
            "jpeg_quality": jpeg_quality,
            "device": device
        }, f, indent=4, ensure_ascii=False)

    # 3. 加载数据与模型
    images_dir = dataset_config["dataset_root"]

    dataset = LocalFlickrDataset(
        images_dir=images_dir,
        ann_file=dataset_config["ann_file"],
        max_samples=dataset_config.get("max_samples", None),
        delimiter='|',
        img_size=img_size,
        whitelist_path=dataset_config.get("whitelist_path")  # 传入参数
    )

    if len(dataset) == 0:
        print("Error: 数据集加载为空。请检查:")
        return

    dataloader = DataLoader(dataset, 
                            batch_size=dataset_config["batch_size"], 
                            shuffle=False, 
                            num_workers=8,
                            pin_memory=True)

    print("\n--- Loading CLIP Model ---")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
    attacker = MultimodalAttacker(model, processor, device)

    final_results = []
    sample_logs = [] # 用于记录具体的 Top-1 文本案例

    print("\n[Phase 1] Evaluating Clean Performance & Building Global Index")
    clean_img_feats_list = []
    text_feats_list = []
    all_img_names = []
    text_image_map = []
    all_texts_raw = [] # 存储原始文本字符串，用于后续检索 Top-1 内容

    for images, texts, img_names in tqdm(dataloader, desc="Extracting Clean Feats"):
        images = images.to(device)
        with torch.no_grad():
            norm_imgs = attacker.normalizer(images)
            img_emb = model.get_image_features(pixel_values=norm_imgs)
            clean_img_feats_list.append(img_emb.cpu())

            txt_emb = attacker.get_text_features(list(texts))
            text_feats_list.append(txt_emb.cpu())

            all_img_names.extend(img_names)
            text_image_map.extend(list(img_names))
            all_texts_raw.extend(list(texts)) # 收集所有文本

    clean_img_feats = torch.cat(clean_img_feats_list, dim=0).to(device)
    text_feats = torch.cat(text_feats_list, dim=0).to(device)
    
    print("[Info] Building Global Ground-Truth Map...")
    gt_map = defaultdict(list)
    for t_idx, imgname in enumerate(text_image_map):
        gt_map[imgname].append(t_idx)

    # Clean Metrics
    clean_metrics = evaluate_global_retrieval(
        clean_img_feats, text_feats, all_img_names, gt_map, k_list=[1,3,5,10], batch_size=256, device=device
    )
    print(f"Clean Metrics: R@1={clean_metrics['R@1']:.2f}%, MR={clean_metrics['Mean Rank']:.2f}")

    final_results.append({
        "Method": "Clean (No Attack)",
        **clean_metrics
    })

    print("\n[Phase 2] Comparative Experiment (Attack & Defense)")

    for config in configurations:
        print(f"\n>>> Mode: {config['name']}")
        adv_img_feats = []
        defended_img_feats = []
        
        # 用于记录当前方法的 Sample 案例
        sample_log_entry = {"Method": config['name']}

        pbar = tqdm(dataloader, desc=f"[{config['name']}] Running")
        for i, (images, texts, img_names) in enumerate(pbar):
            images = images.to(device)
            with torch.no_grad():
                current_text_feats = attacker.get_text_features(list(texts)).to(device)

            # Attack
            adv_images = attacker.mi_fgsm_attack(
                images, current_text_feats,
                epsilon=config["epsilon"], alpha=config["alpha"],
                steps=config["steps"], decay=config["decay"], targeted=False
            )

            # Defense
            defended_images = batch_jpeg_compress_defense(adv_images, quality=jpeg_quality)
            defended_images = defended_images.clamp(0.0, 1.0)

            # Extract Features
            with torch.no_grad():
                norm_adv = attacker.normalizer(adv_images)
                adv_emb = model.get_image_features(pixel_values=norm_adv)
                adv_img_feats.append(adv_emb.cpu())

                norm_def = attacker.normalizer(defended_images)
                def_emb = model.get_image_features(pixel_values=norm_def)
                defended_img_feats.append(def_emb.cpu())
            
            # --- 可视化与 Top-1 文本记录 (仅对每种方法的第一个 Batch 的第一张图做) ---
            if i == 0:
                # 保存可视化图片到实验文件夹
                save_name = os.path.join(exp_dir, f"vis_{config['name']}.png")
                visualize_attack_result(images[0].cpu(), adv_images[0].cpu(), save_name=save_name)
                
                # 计算并记录 Top-1 文本
                # 获取该样本的 Clean Feature
                with torch.no_grad():
                    norm_clean_sample = attacker.normalizer(images[0:1])
                    clean_sample_emb = model.get_image_features(pixel_values=norm_clean_sample)
                
                # 检索 Top-1
                pred_clean = get_top1_text_prediction(clean_sample_emb, text_feats, all_texts_raw)
                pred_adv = get_top1_text_prediction(adv_emb[0:1].to(device), text_feats, all_texts_raw)
                pred_def = get_top1_text_prediction(def_emb[0:1].to(device), text_feats, all_texts_raw)
                
                # 记录到日志字典
                sample_log_entry["Image"] = str(img_names[0])
                sample_log_entry["True_Text"] = str(texts[0])
                sample_log_entry["Original_Top1"] = pred_clean
                sample_log_entry["Attacked_Top1"] = pred_adv
                sample_log_entry["Defended_Top1"] = pred_def
                
                sample_logs.append(sample_log_entry)
                
                print(f"\n[Sample Log] Image: {img_names[0]}")
                print(f"  > Original Top1: {pred_clean[:60]}...")
                print(f"  > Attacked Top1: {pred_adv[:60]}...")

        # 全局评估
        adv_img_feats = torch.cat(adv_img_feats, dim=0).to(device)
        adv_metrics = evaluate_global_retrieval(adv_img_feats, text_feats, all_img_names, gt_map, k_list=[1,3,5,10], batch_size=256, device=device)
        
        final_results.append({"Method": config['name'], **adv_metrics})

        defended_img_feats = torch.cat(defended_img_feats, dim=0).to(device)
        def_metrics = evaluate_global_retrieval(defended_img_feats, text_feats, all_img_names, gt_map, k_list=[1,3,5,10], batch_size=256, device=device)
        
        final_results.append({"Method": config['name'] + " + JPEG", **def_metrics})

    # 4. 实验结束：保存所有结果
    print("\n" + "="*60)
    print("FINAL EXPERIMENTAL RESULTS")
    print("="*60)
    
    # 转换为 DataFrame
    df = pd.DataFrame(final_results)
    cols = ["Method", "R@1", "R@3", "R@5", "R@10", "Mean Rank"]
    print(df[cols].to_markdown(index=False, floatfmt=".2f"))
    
    # 保存结果 CSV
    csv_path = os.path.join(exp_dir, "results.csv")
    df[cols].to_csv(csv_path, index=False)
    print(f"[Saved] Metrics saved to {csv_path}")

    # 保存结果 JSON (包含详细 Sample Log)
    json_result_path = os.path.join(exp_dir, "results.json")
    full_output = {
        "metrics": final_results,
        "sample_cases": sample_logs
    }
    with open(json_result_path, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=4, ensure_ascii=False)
    print(f"[Saved] Detailed JSON results saved to {json_result_path}")

    # 保存 Summary 图片
    summary_plot_path = os.path.join(exp_dir, "experiment_summary.png")
    plot_performance_comparison(final_results, save_name=summary_plot_path)
    print(f"[Saved] Summary chart saved to {summary_plot_path}")


if __name__ == "__main__":
    main()