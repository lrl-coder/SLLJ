import torch
import torch.nn.functional as F


class MultimodalAttacker:
    def __init__(self, model, processor, device="cuda"):
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.model.eval()

        # 定义可微的 Normalize 层
        from dataset import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, DifferentiableNormalize
        self.normalizer = DifferentiableNormalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD).to(device)

    def get_text_features(self, text_list):
        """预计算文本特征，避免循环中重复计算"""
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
            text_embeds = F.normalize(text_embeds, p=2, dim=1)
        return text_embeds

    def mi_fgsm_attack(self, images, target_text_embeds, epsilon=8 / 255, alpha=2 / 255, steps=10, decay=1.0,
                       targeted=False):
        """
        MI-FGSM (Momentum Iterative FGSM) 攻击实现
        Args:
            images: 原始图像 tensor [Batch, 3, H, W], 范围 [0, 1]
            target_text_embeds: 目标文本的特征向量
            epsilon: 最大扰动限制 L_inf
            alpha: 单步步长
            steps: 迭代次数
            decay: 动量衰减因子 (mu), 1.0 表示保持动量
            targeted: Boolean, True=有目标攻击(让图像靠近目标文本), False=无目标攻击(让图像远离真实文本)
        """
        # 1. 初始化对抗样本和动量
        adv_images = images.clone().detach()
        adv_images.requires_grad = True
        momentum = torch.zeros_like(images).to(self.device)

        print(f"  [Attack] Mode: {'Targeted' if targeted else 'Untargeted'} | MI-FGSM Steps: {steps}")

        for step in range(steps):
            # 2. 动态构建输入：先 Normalize 再送入模型
            # 这一步至关重要，因为攻击是在 [0,1] 空间，但模型需要 Normalized 输入
            norm_adv_images = self.normalizer(adv_images)

            # 3. 获取图像特征
            image_embeds = self.model.get_image_features(pixel_values=norm_adv_images)
            image_embeds = F.normalize(image_embeds, p=2, dim=1)

            # 4. 计算损失：Cosine Similarity
            # loss = sum(img_emb * text_emb)
            similarity = torch.sum(image_embeds * target_text_embeds)

            # 5. 定义优化目标
            if targeted:
                # Targeted: 我们希望 Similarity 越大越好 -> Loss 越小越好 -> Loss = -Similarity
                loss = -similarity
            else:
                # Untargeted: 我们希望 Similarity 越小越好 (远离原义) -> Loss = Similarity
                loss = similarity

            # 6. 反向传播
            self.model.zero_grad()
            loss.backward()

            # 7. 计算动量梯度 (MI-FGSM 核心)
            grad = adv_images.grad.data
            # L1 归一化梯度，保证梯度的尺度稳定性
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            momentum = decay * momentum + grad

            # 8. 更新图像
            # update = alpha * sign(momentum)
            # 因为我们要 minimize Loss，所以是减去梯度
            adv_images.data = adv_images.data - alpha * torch.sign(momentum)

            # 9. 投影 (Projection) 回 L_inf 球
            delta = torch.clamp(adv_images.data - images.data, -epsilon, epsilon)
            adv_images.data = torch.clamp(images.data + delta, 0.0, 1.0)

            adv_images.grad.zero_()

        return adv_images.detach()