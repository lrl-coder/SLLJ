import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import torch.nn.functional as F


def setup_seed(seed=42):
    """
    锁定所有随机种子，保证实验结果可复现。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[Utils] Random seed set to {seed}")


def tensor_to_numpy(tensor):
    """
    辅助函数：将 PyTorch Tensor 转换为可显示的 Numpy 数组。
    Args:
        tensor: [C, H, W], 范围 [0, 1] 或任意
    Returns:
        numpy array: [H, W, C], 范围 [0, 1]
    """
    # 确保 tensor 在 CPU 上，并去掉 batch 维度（如果有）
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # 交换维度 C,H,W -> H,W,C
    img_np = tensor.permute(1, 2, 0).detach().cpu().numpy()

    # 裁剪到合法范围，防止显示异常
    img_np = np.clip(img_np, 0.0, 1.0)
    return img_np


def visualize_attack_result(clean_img, adv_img, save_name="attack_result.png"):
    """
    可视化攻击结果：原图 vs 扰动噪声 vs 对抗样本
    Args:
        clean_img: 原始图像 Tensor (C, H, W)
        adv_img: 对抗图像 Tensor (C, H, W)
        save_name: 保存文件名
    """
    # 计算扰动噪声
    # 这里的 50 是放大倍数，为了让肉眼能看清微小的噪声模式
    perturbation = (adv_img - clean_img) * 50 + 0.5

    # 转换为 Numpy 格式
    np_clean = tensor_to_numpy(clean_img)
    np_noise = tensor_to_numpy(perturbation)
    np_adv = tensor_to_numpy(adv_img)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np_clean)
    axes[0].set_title("Original Image (Clean)", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(np_noise)
    axes[1].set_title("Perturbation (x50 amplified)", fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(np_adv)
    axes[2].set_title("Adversarial Image", fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()

    # 加上 bbox_inches='tight' 和 pad_inches
    # bbox_inches='tight' 会自动计算包含所有文字的最小边框，确保不被裁剪
    plt.savefig(save_name, dpi=150, bbox_inches='tight', pad_inches=0.1)
    print(f"[Utils] Visualization saved to {save_name}")
    plt.close()


def calculate_recall_metrics(image_embeds, text_embeds, ground_truth_indices, k_list=[1, 5, 10]):
    """
    计算图像到文本检索的 Recall@K 指标。

    Args:
        image_embeds: [N, D] 图像特征矩阵
        text_embeds: [M, D] 文本特征矩阵 (通常 M >= N)
        ground_truth_indices: [N] 每个图像对应的正确文本在 text_embeds 中的索引
        k_list: 需要计算的 K 值列表

    Returns:
        dict: 包含各个 K 值的 Recall 分数
    """
    # 确保特征已经归一化
    image_embeds = F.normalize(image_embeds, p=2, dim=1)
    text_embeds = F.normalize(text_embeds, p=2, dim=1)

    # 计算相似度矩阵 [N, M]
    sim_matrix = torch.matmul(image_embeds, text_embeds.t())

    results = {}
    num_images = image_embeds.shape[0]

    # 获取每一行相似度最高的 Top-K 索引
    # max_k 是我们需要评估的最大 k 值
    max_k = max(k_list)
    _, top_indices = sim_matrix.topk(max_k, dim=1)  # [N, max_k]

    top_indices = top_indices.cpu()
    ground_truth_indices = ground_truth_indices.cpu()

    for k in k_list:
        # 取前 k 个预测
        curr_top_k = top_indices[:, :k]

        # 检查 ground_truth 是否在 curr_top_k 中
        # any(dim=1) 检查每一行是否有匹配
        hits = (curr_top_k == ground_truth_indices.view(-1, 1)).any(dim=1)
        recall = hits.sum().item() / num_images
        results[f"R@{k}"] = recall

    return results