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
from tqdm import tqdm
from collections import defaultdict

# 导入模块
from utils.dataset import LocalFlickrDataset
from utils.attacker import MultimodalAttacker
from utils.utils import setup_seed, evaluate_global_retrieval
from config.algorithm_config import configurations
from config.transfer_dataset_config import dataset_config

torch.backends.cudnn.benchmark = True



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model", type=str, default="clip-vit-base-patch16", help="Source model")
    parser.add_argument("--target_model", type=str, default="clip-vit-large-patch14", help="Target model")
    args = parser.parse_args()
    # source and target model id
    SOURCE_MODEL_ID = f'openai/{args.source_model}'
    TARGET_MODEL_ID = f'openai/{args.target_model}'
    # 1. 初始化
    setup_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join("results", f"transfer_exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"--- Transferability Experiment Results will be saved to: {exp_dir} ---")
    
    # 确保分辨率一致 (默认 224)
    img_size = 224 
    if "336" in SOURCE_MODEL_ID or "336" in TARGET_MODEL_ID:
        print("[Warning] Source 或 Target 包含 336px 模型，直接像素迁移可能导致尺寸不匹配或插值失真。建议使用同分辨率模型。")
        # 如果必须跑，这里需要逻辑处理 resize，本脚本暂定为 224

    # 保存配置
    with open(os.path.join(exp_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "source_model": SOURCE_MODEL_ID,
            "target_model": TARGET_MODEL_ID,
            "attack_configs": configurations,
            "dataset": dataset_config
        }, f, indent=4)

    # 2. 加载数据集
    images_dir = dataset_config["dataset_root"]
    dataset = LocalFlickrDataset(
        images_dir=images_dir,
        ann_file=dataset_config["ann_file"],
        max_samples=dataset_config.get("max_samples", None),
        delimiter='|',
        img_size=img_size,
        whitelist_path=dataset_config.get("whitelist_path")
    )
    
    dataloader = DataLoader(dataset, batch_size=dataset_config["batch_size"], 
                            shuffle=False, num_workers=4, pin_memory=True)

    # 3. 加载模型
    print(f"\n[Load] Source Model: {SOURCE_MODEL_ID}")
    src_model = CLIPModel.from_pretrained(SOURCE_MODEL_ID).to(device)
    src_processor = CLIPProcessor.from_pretrained(SOURCE_MODEL_ID, use_fast=True)
    attacker_source = MultimodalAttacker(src_model, src_processor, device)

    print(f"[Load] Target Model: {TARGET_MODEL_ID}")
    tgt_model = CLIPModel.from_pretrained(TARGET_MODEL_ID).to(device)
    tgt_processor = CLIPProcessor.from_pretrained(TARGET_MODEL_ID, use_fast=True)
    # Target 不需要用来攻击，但我们可以复用 Attacker 类里的 normalizer 和 feature extraction 功能
    attacker_target = MultimodalAttacker(tgt_model, tgt_processor, device)

    final_results = []
    
    # --- 阶段 A: 准备 Target 模型的文本索引库 ---
    # 评估迁移性时，我们是在 Target 模型上做检索，所以必须用 Target 模型的 Text Encoder 建立索引
    print("\n[Phase A] Extracting Target Model Text Features (The Database)...")
    target_text_feats_list = []
    all_img_names = []
    text_image_map = []
    
    # 只需要跑一遍数据来提取 Target 文本特征 和 建立 GT Map
    for _, texts, img_names in tqdm(dataloader, desc="Target Text Indexing"):
        with torch.no_grad():
            # 使用 Target 模型的 Tokenizer 和 Encoder
            txt_emb = attacker_target.get_text_features(list(texts))
            target_text_feats_list.append(txt_emb.cpu())
            
        all_img_names.extend(img_names)
        text_image_map.extend(list(img_names))

    target_text_feats = torch.cat(target_text_feats_list, dim=0).to(device)
    
    # 建立 Ground Truth Map
    gt_map = defaultdict(list)
    for t_idx, imgname in enumerate(text_image_map):
        gt_map[imgname].append(t_idx)

    # --- 阶段 B: 评估 Clean Images 在 Target 模型上的表现 (Baseline) ---
    print("\n[Phase B] Evaluating Clean Baseline on Target Model...")
    clean_tgt_img_feats = []
    for images, _, _ in tqdm(dataloader, desc="Clean Target Feats"):
        images = images.to(device)
        with torch.no_grad():
            norm_imgs = attacker_target.normalizer(images)
            img_emb = tgt_model.get_image_features(pixel_values=norm_imgs)
            clean_tgt_img_feats.append(img_emb.cpu())
            
    clean_tgt_img_feats = torch.cat(clean_tgt_img_feats, dim=0).to(device)
    clean_metrics = evaluate_global_retrieval(
        clean_tgt_img_feats, target_text_feats, all_img_names, gt_map, k_list=[1,5,10], device=device
    )
    print(f"Target Clean Metrics: R@1={clean_metrics['R@1']:.2f}%")
    final_results.append({
        "Attack Method": "Clean (No Attack)",
        "Source Model": "N/A", 
        "Target Model": TARGET_MODEL_ID,
        **clean_metrics
    })

    # --- 阶段 C: 生成对抗样本并迁移攻击 ---
    print("\n[Phase C] Running Transfer Attacks...")
    
    for config in configurations:
        print(f"\n>>> Generating Attacks using {config['name']} on Source Model...")
        
        adv_tgt_img_feats = [] # 存储对抗样本在 Target 模型上的特征
        
        pbar = tqdm(dataloader, desc=f"Transfer: Source->Target")
        for images, texts, _ in pbar:
            images = images.to(device)
            
            # 1. 在 Source 模型上生成对抗样本
            # 需要 Source 模型的 Text Features 作为攻击引导
            with torch.no_grad():
                src_text_feats = attacker_source.get_text_features(list(texts)).to(device)
            
            # 生成攻击 (White-box on Source)
            adv_images = attacker_source.mi_fgsm_attack(
                images, src_text_feats,
                epsilon=config["epsilon"], alpha=config["alpha"],
                steps=config["steps"], decay=config["decay"], targeted=False
            )
            
            # 2. 将生成的对抗样本输入 Target 模型提取特征
            # 注意：这里模拟黑盒，直接把图喂给 Target
            with torch.no_grad():
                norm_adv = attacker_target.normalizer(adv_images)
                adv_emb_tgt = tgt_model.get_image_features(pixel_values=norm_adv)
                adv_tgt_img_feats.append(adv_emb_tgt.cpu())

        # 3. 评估迁移攻击效果
        adv_tgt_img_feats = torch.cat(adv_tgt_img_feats, dim=0).to(device)
        transfer_metrics = evaluate_global_retrieval(
            adv_tgt_img_feats, target_text_feats, all_img_names, gt_map, k_list=[1,5,10], device=device
        )
        
        print(f"[{config['name']}] Transfer Result on Target: R@1={transfer_metrics['R@1']:.2f}%")
        
        final_results.append({
            "Attack Method": config['name'],
            "Source Model": SOURCE_MODEL_ID,
            "Target Model": TARGET_MODEL_ID,
            **transfer_metrics
        })

    # 4. 保存结果
    print("\n" + "="*60)
    print("TRANSFERABILITY EXPERIMENT RESULTS")
    print("="*60)
    
    df = pd.DataFrame(final_results)
    cols = ["Attack Method", "Source Model", "Target Model", "R@1", "R@5", "R@10", "Mean Rank"]
    print(df[cols].to_markdown(index=False, floatfmt=".2f"))
    
    csv_path = os.path.join(exp_dir, "transfer_results.csv")
    df[cols].to_csv(csv_path, index=False)
    print(f"\n[Saved] Results saved to {csv_path}")

if __name__ == "__main__":
    main()