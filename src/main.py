import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import matplotlib
from tqdm import tqdm
from collections import defaultdict

matplotlib.use('Agg')

from dataset import LocalFlickrDataset
from attacker import MultimodalAttacker
# 注意这里引入的是更新后的 evaluate_global_retrieval
from utils import setup_seed, visualize_attack_result, evaluate_global_retrieval, plot_performance_comparison
from defence import jpeg_compress_defense

def main():
    setup_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    dataset_config = {
        "dataset_root": r"../../dataset/flickr30k_images",
        "ann_file": r"../../dataset/flickr30k_images/results.csv",
        "max_samples": 1000,
        "batch_size": 32,
        # 修改：禁用 interim 采样，确保每个 batch 都对全量文本库进行检索评估
        "use_sampling_for_interim": False 
    }

    configurations = [
        {"name": "FGSM", "epsilon": 8/255, "alpha": 8/255, "steps": 1, "decay": 0.0},
        {"name": "PGD", "epsilon": 8/255, "alpha": 2/255, "steps": 10, "decay": 0.0},
        {"name": "MI-FGSM", "epsilon": 8/255, "alpha": 2/255, "steps": 10, "decay": 1.0}
    ]

    images_dir = os.path.join(dataset_config["dataset_root"], "flickr30k_images")
    dataset = LocalFlickrDataset(
        images_dir=images_dir,
        ann_file=dataset_config["ann_file"],
        max_samples=dataset_config.get("max_samples", None),
        delimiter='|'
    )

    if len(dataset) == 0:
        print("Error: 数据集加载为空，请检查路径。")
        return

    dataloader = DataLoader(dataset, batch_size=dataset_config["batch_size"], shuffle=False, num_workers=0)

    print("\n--- Loading CLIP Model ---")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    attacker = MultimodalAttacker(model, processor, device)

    final_results = []

    print("\n[Phase 1] Evaluating Clean Performance & Building Global Index")
    clean_img_feats_list = []
    text_feats_list = []
    all_img_names = []
    text_image_map = []  # List[str]: 记录每个文本特征向量对应的 image_name

    for images, texts, img_names in tqdm(dataloader, desc="Extracting Clean Feats"):
        images = images.to(device)
        with torch.no_grad():
            norm_imgs = attacker.normalizer(images)
            img_emb = model.get_image_features(pixel_values=norm_imgs)
            clean_img_feats_list.append(img_emb.cpu())

            # 提取文本特征
            txt_emb = attacker.get_text_features(list(texts))
            text_feats_list.append(txt_emb.cpu())

            # 记录用于后续匹配的元数据
            all_img_names.extend(img_names)
            text_image_map.extend(list(img_names))

    clean_img_feats = torch.cat(clean_img_feats_list, dim=0).to(device)
    text_feats = torch.cat(text_feats_list, dim=0).to(device)
    
    # === 关键步骤：预构建全局 Ground Truth 映射 ===
    # 将 list 转换为 dict: {image_name: [text_idx1, text_idx2, ...]}
    # 这样在评估时，只需 O(1) 就能获取一张图片对应的所有正确文本索引
    print("[Info] Building Global Ground-Truth Map...")
    gt_map = defaultdict(list)
    for t_idx, imgname in enumerate(text_image_map):
        gt_map[imgname].append(t_idx)

    # 计算 Clean 全局指标
    clean_metrics = evaluate_global_retrieval(
        clean_img_feats, text_feats, all_img_names, gt_map, k_list=[1,3,5,10], batch_size=256, device=device
    )
    print(f"Clean Metrics: R@1={clean_metrics['R@1']:.2f}%, MR={clean_metrics['Mean Rank']:.2f}")

    final_results.append({
        "Method": "Clean (No Attack)",
        "R@1": clean_metrics['R@1'],
        "R@3": clean_metrics['R@3'],
        "R@5": clean_metrics['R@5'],
        "R@10": clean_metrics['R@10'],
        "Mean Rank": clean_metrics['Mean Rank']
    })

    print("\n[Phase 2] Comparative Experiment (Attack & Defense)")

    for config in configurations:
        print(f"\n>>> Mode: {config['name']}")
        adv_img_feats = []
        defended_img_feats = []

        pbar = tqdm(dataloader, desc=f"[{config['name']}] Running")
        for i, (images, texts, img_names) in enumerate(pbar):
            images = images.to(device)
            with torch.no_grad():
                current_text_feats = attacker.get_text_features(list(texts)).to(device)

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

            # 2. 防御：JPEG 压缩
            defended_images_batch = []
            for j in range(adv_images.size(0)):
                def_img = jpeg_compress_defense(adv_images[j], quality=50)
                defended_images_batch.append(def_img)
            defended_images = torch.stack(defended_images_batch).to(device)
            defended_images = defended_images.clamp(0.0, 1.0)

            # 3. 特征提取
            with torch.no_grad():
                norm_adv = attacker.normalizer(adv_images)
                adv_emb = model.get_image_features(pixel_values=norm_adv)
                adv_img_feats.append(adv_emb.cpu())

                norm_def = attacker.normalizer(defended_images)
                def_emb = model.get_image_features(pixel_values=norm_def)
                defended_img_feats.append(def_emb.cpu())

                # --- 实时评估 (Interim Evaluation) ---
                # 使用当前 batch 的图片，去检索【全局】文本库 (text_feats)
                # 并使用预计算的 gt_map 判定正确性
                # 耗时！！
                # batch_adv_metrics = evaluate_global_retrieval(
                #     adv_emb.to(device), 
                #     text_feats,     # 全局文本
                #     list(img_names),# 当前 batch 图片名
                #     gt_map,         # 全局 GT 映射
                #     k_list=[1], 
                #     batch_size=adv_emb.size(0), 
                #     device=device
                # )
                
                # batch_def_metrics = evaluate_global_retrieval(
                #     def_emb.to(device), 
                #     text_feats, 
                #     list(img_names), 
                #     gt_map, 
                #     k_list=[1], 
                #     batch_size=def_emb.size(0), 
                #     device=device
                # )

                # pbar.set_postfix({
                #     "Adv_R1": f"{batch_adv_metrics['R@1']:.1f}%", 
                #     "Def_R1": f"{batch_def_metrics['R@1']:.1f}%"
                # })

            if i == 0:
                os.makedirs("out", exist_ok=True)
                save_name = f"out/vis_{config['name']}.png"
                visualize_attack_result(images[0].cpu(), adv_images[0].cpu(), save_name=save_name)

        # 全局评估（合并所有 batch 的特征）
        # 虽然 batch 内部已经是全局检索，这里为了统计 R@3/5/10 仍然做一次汇总
        adv_img_feats = torch.cat(adv_img_feats, dim=0).to(device)
        adv_metrics = evaluate_global_retrieval(adv_img_feats, text_feats, all_img_names, gt_map, k_list=[1,3,5,10], batch_size=256, device=device)
        print(f"[{config['name']}] FINAL Attack R@1: {adv_metrics['R@1']:.2f}%")

        res_dict = {"Method": config['name']}
        res_dict.update(adv_metrics)
        final_results.append(res_dict)

        defended_img_feats = torch.cat(defended_img_feats, dim=0).to(device)
        def_metrics = evaluate_global_retrieval(defended_img_feats, text_feats, all_img_names, gt_map, k_list=[1,3,5,10], batch_size=256, device=device)
        print(f"[{config['name']} + JPEG] FINAL Defense R@1: {def_metrics['R@1']:.2f}%")

        res_dict_def = {"Method": config['name'] + " + JPEG"}
        res_dict_def.update(def_metrics)
        final_results.append(res_dict_def)

    print("\n" + "="*60)
    print("FINAL EXPERIMENTAL RESULTS")
    print("="*60)
    df = pd.DataFrame(final_results)
    cols = ["Method", "R@1", "R@3", "R@5", "R@10", "Mean Rank"]
    print(df[cols].to_markdown(index=False, floatfmt=".2f"))
    print("="*60)

    plot_performance_comparison(final_results, save_name="experiment_summary.png")
    print("\n[Done] Summary chart saved.")

if __name__ == "__main__":
    main()