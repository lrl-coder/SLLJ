import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import torch.nn.functional as F
from collections import defaultdict

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[Utils] Random seed set to {seed}")

def tensor_to_numpy(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0.0, 1.0)
    return img_np

def visualize_attack_result(clean_img, adv_img, save_name="attack_result.png"):
    perturbation = (adv_img - clean_img) * 50 + 0.5
    np_clean = tensor_to_numpy(clean_img)
    np_noise = tensor_to_numpy(perturbation)
    np_adv = tensor_to_numpy(adv_img)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np_clean); axes[0].set_title("Original Image (Clean)"); axes[0].axis('off')
    axes[1].imshow(np_noise); axes[1].set_title("Perturbation (x50)"); axes[1].axis('off')
    axes[2].imshow(np_adv); axes[2].set_title("Adversarial Image"); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight', pad_inches=0.1)
    print(f"[Utils] Visualization saved to {save_name}")
    plt.close()

def evaluate_global_retrieval(image_embeds, text_embeds, img_names, gt_map, k_list=[1,3,5,10], batch_size=128, device=None):
    """
    计算全局检索指标 (Recall@K, Mean Rank) - 使用全排 (argsort) 以获取精确 Rank
    
    Args:
        image_embeds: [N_images, D] 当前 batch 或 全量的图像特征
        text_embeds:  [N_texts, D] 全局文本库特征
        img_names:    list[str] 当前 image_embeds 对应的图片文件名
        gt_map:       dict {image_name: [text_index_1, ...]} 
        k_list:       list[int]
    """
    if device is None:
        device = image_embeds.device
        
    image_embeds = F.normalize(image_embeds, p=2, dim=1).to(device)
    text_embeds = F.normalize(text_embeds, p=2, dim=1).to(device)

    num_images = image_embeds.shape[0]
    num_texts = text_embeds.shape[0]

    recall_counts = {k: 0 for k in k_list}
    total_rank = 0.0

    # 转置文本特征以进行矩阵乘法 [D, N_texts]
    text_emb_t = text_embeds.t()

    for start in range(0, num_images, batch_size):
        end = min(start + batch_size, num_images)
        img_batch = image_embeds[start:end]      # [B, D]
        batch_names = img_names[start:end]       # [B]

        # 1. 计算相似度: [B, N_texts]
        sim = torch.matmul(img_batch, text_emb_t)

        # 2. 全量排序 (Full Sort)
        # 使用 argsort 获取所有文本的排名索引（从高相似度到低相似度）
        # 返回 shape: [B, N_texts]
        # 注意：这里会比 topk 慢，因为要对所有 15w+ 文本进行排序
        sorted_idx = torch.argsort(sim, dim=1, descending=True).cpu().numpy()

        # 3. 逐个样本计算指标
        for i, imgname in enumerate(batch_names):
            gt_indices = gt_map.get(imgname, [])
            
            if not gt_indices:
                total_rank += num_texts
                continue

            # 使用 set 加速 Python 循环中的查找 (如果需要)
            gt_set = set(gt_indices)
            current_sorted = sorted_idx[i]  # [N_texts]

            # --- 计算 Recall@K ---
            for k in k_list:
                # 检查前 k 个结果中是否有任意一个在 gt_indices 中
                topk_slice = current_sorted[:k]
                # 使用 set intersection check
                if not gt_set.isdisjoint(topk_slice):
                    recall_counts[k] += 1

            # --- 计算精确 Mean Rank ---
            # 目标：找到 sorted_idx 中 *第一个* 出现的 gt_index 的位置 (1-based)
            
            # 方法：利用 numpy 的向量化操作快速定位
            # isin 返回一个布尔数组，表示 current_sorted 中每个元素是否在 gt_indices 里
            hits = np.isin(current_sorted, gt_indices)
            
            if hits.any():
                # argmax 会返回第一个 True 的索引位置
                first_hit_index = np.argmax(hits)
                rank = first_hit_index + 1
            else:
                # 理论上不应该发生，除非 gt_map 和 text_embeds 不匹配
                rank = num_texts
            
            total_rank += rank

    metrics = {}
    for k in k_list:
        metrics[f"R@{k}"] = (recall_counts[k] / num_images) * 100.0
    metrics["Mean Rank"] = total_rank / num_images
    return metrics

def plot_performance_comparison(final_results, save_name="experiment_summary.png"):
    methods = []
    r1_attack = []
    r1_defense = []
    res_dict = {item['Method']: item['R@1'] for item in final_results}
    
    seen_methods = set()
    for item in final_results:
        name = item['Method']
        base_name = name.replace(" + JPEG", "")
        if base_name not in seen_methods:
            seen_methods.add(base_name)
            methods.append(base_name)
            r1_attack.append(res_dict.get(base_name, 0.0))
            r1_defense.append(res_dict.get(base_name + " + JPEG", 0.0))

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, r1_attack, width, label='No Defense')
    rects2 = ax.bar(x + width/2, r1_defense, width, label='With JPEG Defense')
    ax.set_ylabel('Recall@1 (%)')
    ax.set_title('Attack Impact & Defense Effectiveness')
    ax.set_xticks(x); ax.set_xticklabels(methods); ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.bar_label(rects1, padding=3, fmt='%.1f'); ax.bar_label(rects2, padding=3, fmt='%.1f')
    fig.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"[Utils] Performance chart saved to {save_name}")
    plt.close()