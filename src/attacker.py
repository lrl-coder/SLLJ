import torch
import torch.nn.functional as F
from dataset import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, DifferentiableNormalize


class MultimodalAttacker:
    def __init__(self, model, processor, device="cuda"):
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.model.eval()

        # 冻结模型参数，确保只对图像求导
        for param in self.model.parameters():
            param.requires_grad = False

        # 初始化可微归一化层 (适配 CLIP 的预处理)
        self.normalizer = DifferentiableNormalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD).to(device)

    def get_text_features(self, text_list):
        """预计算文本特征，并进行归一化"""
        # truncation=True 防止文本过长报错
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
            text_embeds = F.normalize(text_embeds, p=2, dim=1)
        return text_embeds

    def mi_fgsm_attack(self, images, target_text_embeds, epsilon=8 / 255, alpha=2 / 255, steps=10, decay=1.0,
                       targeted=False):
        """
        通用对抗攻击方法，可通过参数配置实现 FGSM, PGD, MI-FGSM。

        Args:
            images: 原始图像 (Batch, C, H, W), 范围 [0, 1]
            target_text_embeds: 目标文本特征 (Batch, D)
            epsilon: 最大扰动限制 (L_inf norm)
            alpha: 单步步长
            steps: 迭代次数 (FGSM设为1, PGD/MI-FGSM设为10+)
            decay: 动量衰减因子 (FGSM/PGD设为0, MI-FGSM设为1.0)
            targeted: True=定向攻击(靠近目标), False=无定向攻击(远离目标)
        """
        # 1. 初始化
        adv_images = images.clone().detach()
        adv_images.requires_grad = True
        momentum = torch.zeros_like(images).to(self.device)

        # 2. 迭代攻击
        for step in range(steps):
            # 每次迭代前先清空梯度
            if adv_images.grad is not None:
                adv_images.grad.zero_()

            # 前向传播：先 Normalize 再送入模型
            # 攻击是在 [0,1] 空间进行的，输入模型前需归一化
            norm_adv_images = self.normalizer(adv_images)
            image_embeds = self.model.get_image_features(pixel_values=norm_adv_images)
            image_embeds = F.normalize(image_embeds, p=2, dim=1)

            # 计算余弦相似度损失: Sim(Img, Text)
            # target_text_embeds 必须与 images 一一对应
            similarity = torch.sum(image_embeds * target_text_embeds, dim=1)

            # 定义损失函数
            # Targeted: Minimize (-Similarity) => Maximize Similarity (让图像靠近目标文本)
            # Untargeted: Minimize (Similarity) => Maximize Distance (让图像远离目标文本)
            if targeted:
                loss = -torch.mean(similarity)
            else:
                loss = torch.mean(similarity)

            # 反向传播计算梯度
            loss.backward()

            # 获取梯度
            grad = adv_images.grad.data

            # 3. 计算动量 (MI-FGSM 核心)
            # 梯度 L1 归一化，保证动量累积的尺度稳定性
            grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad / (grad_norm + 1e-12)  # 加上微小值防止除零
            momentum = decay * momentum + grad

            # 4. 更新图像
            # 总是沿着损失下降的方向更新
            adv_images.data = adv_images.data - alpha * torch.sign(momentum)

            # 5. 投影 (Projection)
            # 限制扰动在 epsilon 球内
            delta = torch.clamp(adv_images.data - images.data, -epsilon, epsilon)
            # 限制图像像素在 [0, 1] 范围内
            adv_images.data = torch.clamp(images.data + delta, 0.0, 1.0)

            # 重新开启梯度记录供下一次迭代使用
            adv_images.requires_grad = True

        return adv_images.detach()