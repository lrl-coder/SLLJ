import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np


# ==========================================
# 1. 配置与工具类
# ==========================================

class AttackConfig:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "openai/clip-vit-base-patch32"
        # PGD 攻击超参数
        self.epsilon = 8 / 255  # 最大扰动幅度 (L_inf norm)
        self.alpha = 2 / 255  # 单步更新步长
        self.steps = 10  # 迭代次数
        self.batch_size = 1


def load_demo_image():
    """从网络加载一张测试图片，模拟数据加载"""
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


# ==========================================
# 2. 核心攻击类：Projected Gradient Descent
# ==========================================

class MultimodalAttacker:
    def __init__(self, model, processor, config):
        self.model = model.to(config.device)
        self.processor = processor
        self.config = config
        self.device = config.device

        # 冻结模型参数，我们只攻击输入，不训练模型
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def pgd_attack(self, image, true_text):
        """
        执行 PGD 攻击
        Args:
            image: PIL Image 对象
            true_text: 对应的真实文本描述
        Returns:
            adv_image_tensor: 对抗样本张量
        """
        # 1. 预处理：将图片转换为 Tensor，但先不进行 Normalize（归一化）
        # CLIPProcessor 通常会做 Resize -> Crop -> Rescale(0-1) -> Normalize
        # 为了方便计算梯度，我们手动处理 Tensor 转换
        inputs = self.processor(text=[true_text], images=image, return_tensors="pt", padding=True)

        # 提取 clean image tensor (Batch, Channel, Height, Width)
        # 注意：这里的 pixel_values 已经被 processor 归一化了。
        # 为了更精准控制，理想情况下应该在 [0,1] 空间攻击，再手动归一化。
        # 这里为了简化代码，我们直接在 normalized 空间进行攻击演示，但会限制扰动范围。
        clean_images = inputs['pixel_values'].to(self.device)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # 复制一份图像用于添加扰动，并开启梯度记录
        adv_images = clean_images.clone().detach()
        adv_images.requires_grad = True

        print(f"Starting PGD Attack (Steps: {self.config.steps})...")

        for step in range(self.config.steps):
            # -----------------------------------------------------------
            # 关键步骤 A: 前向传播
            # -----------------------------------------------------------
            outputs = self.model(pixel_values=adv_images,
                                 input_ids=input_ids,
                                 attention_mask=attention_mask)

            # 获取图像和文本的 Embedding
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # -----------------------------------------------------------
            # 关键步骤 B: 定义损失函数
            # 目标：让图像和文本越不相似越好
            # -----------------------------------------------------------
            # 计算余弦相似度 (Cosine Similarity)
            # 需要先对特征向量进行 L2 归一化
            image_embeds_norm = F.normalize(image_embeds, p=2, dim=1)
            text_embeds_norm = F.normalize(text_embeds, p=2, dim=1)

            # 计算相似度得分 (Batch size 为 1 时，结果是一个标量)
            similarity = torch.sum(image_embeds_norm * text_embeds_norm)

            # 我们希望最小化相似度，所以 Loss = Similarity
            # 梯度下降会最小化 Loss，从而降低相似度
            loss = similarity

            # -----------------------------------------------------------
            # 关键步骤 C: 反向传播与梯度计算
            # -----------------------------------------------------------
            self.model.zero_grad()
            loss.backward()  # 计算 loss 对 adv_images 的梯度

            # 获取梯度方向 (Sign Method)
            data_grad = adv_images.grad.data

            # -----------------------------------------------------------
            # 关键步骤 D: 更新像素
            # 公式: x_adv = x_adv - alpha * sign(gradient)
            # (减号是因为我们要最小化相似度)
            # -----------------------------------------------------------
            adv_images.data = adv_images.data - self.config.alpha * data_grad.sign()

            # -----------------------------------------------------------
            # 关键步骤 E: 投影 (Projection) 与 裁剪 (Clamping)
            # 保证扰动不超过 epsilon，且图像数值有效
            # -----------------------------------------------------------
            # 1. 计算总扰动
            eta = torch.clamp(adv_images.data - clean_images.data, -self.config.epsilon, self.config.epsilon)
            # 2. 应用扰动
            adv_images.data = clean_images.data + eta

            # 清空梯度，为下一步做准备
            adv_images.grad.zero_()

            # (可选) 打印每一步的相似度变化
            if step % 2 == 0:
                print(f"  Step {step}: Similarity = {loss.item():.4f}")

        return adv_images.detach(), clean_images.detach()


# ==========================================
# 3. 评估模块
# ==========================================

def evaluate_retrieval(model, processor, image_tensor, true_text, distraction_texts, device):
    """
    模拟检索场景：计算图片与一组文本的匹配分数
    """
    all_texts = [true_text] + distraction_texts

    # 处理文本
    text_inputs = processor(text=all_texts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        # 获取特征
        image_embeds = model.get_image_features(pixel_values=image_tensor)
        text_embeds = model.get_text_features(**text_inputs)

        # 归一化
        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)

        # 计算相似度矩阵 (1 x N_texts)
        logits_per_image = torch.matmul(image_embeds, text_embeds.t())
        probs = logits_per_image.softmax(dim=1)  # 转换为概率

    scores = probs.cpu().numpy()[0]

    # 排序：获取概率最高的文本索引
    ranked_indices = np.argsort(scores)[::-1]

    # 检查真实文本（索引0）排在第几位
    true_text_rank = list(ranked_indices).index(0) + 1  # Rank从1开始

    return scores, true_text_rank, all_texts


# ==========================================
# 4. 主程序运行
# ==========================================

if __name__ == "__main__":
    # A. 初始化
    cfg = AttackConfig()
    print(f"Loading CLIP model: {cfg.model_id}...")
    model = CLIPModel.from_pretrained(cfg.model_id)
    processor = CLIPProcessor.from_pretrained(cfg.model_id)
    attacker = MultimodalAttacker(model, processor, cfg)

    # B. 准备数据
    # 这里我们用一只猫的图，但文本是 "two cats"，看看能不能攻击成其他
    image = load_demo_image()  # 这是一张两只猫的图
    true_text = "two cats sleeping on a pink blanket"

    # 干扰文本库（模拟检索库中的其他条目）
    distraction_texts = [
        "a dog running in the park",
        "a delicious pizza",
        "a computer screen with code",
        "a car driving on the highway",
        "a view of the mountains",
        "someone playing basketball"
    ]

    print(f"\nTarget Text: '{true_text}'")

    # C. 攻击前评估 (Baseline)
    print("\n--- Evaluation Before Attack ---")
    inputs = processor(images=image, return_tensors="pt")
    clean_tensor = inputs['pixel_values'].to(cfg.device)
    scores, rank, texts = evaluate_retrieval(model, processor, clean_tensor, true_text, distraction_texts, cfg.device)
    print(f"True Text Rank: {rank} (Score: {scores[0]:.4f})")
    print(f"Top 1 Prediction: {texts[np.argmax(scores)]}")

    # D. 执行攻击
    print("\n--- Running PGD Attack ---")
    adv_tensor, _ = attacker.pgd_attack(image, true_text)

    # E. 攻击后评估
    print("\n--- Evaluation After Attack ---")
    scores_adv, rank_adv, texts_adv = evaluate_retrieval(model, processor, adv_tensor, true_text, distraction_texts,
                                                         cfg.device)

    print(f"True Text Rank: {rank_adv} (Score: {scores_adv[0]:.4f})")
    print(f"Top 1 Prediction: {texts_adv[np.argmax(scores_adv)]}")

    if rank_adv > 1:
        print("\n[SUCCESS] Attack Successful! The model no longer retrieves the correct text first.")
    else:
        print("\n[FAIL] Attack Failed. Try increasing epsilon or steps.")

    # 可选：保存对抗样本用于观察
    # 注意：这里保存的是 Normalize 后的 tensor，可视化需要反归一化