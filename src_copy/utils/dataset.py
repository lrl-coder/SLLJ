import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import pandas as pd


# CLIP 的标准预处理参数
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class MiniCOCODataset(Dataset):
    """
    模拟一个微型数据集，支持返回 (image, text) 对。
    """

    def __init__(self, image_urls, captions, model_name="openai/clip-vit-base-patch32"):
        self.image_urls = image_urls
        self.captions = captions

        # 预处理：只做 Resize 和 ToTensor，不做 Normalize
        # 因为我们需要在 [0,1] 空间加噪声，Normalize 放在模型前向传播里做
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor()
        ])

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, idx):
        url = self.image_urls[idx]
        text = self.captions[idx]

        # 实际项目中建议本地加载，这里为了演示方便用网络请求
        try:
            resp = requests.get(url, timeout=5)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        except:
            img = Image.new('RGB', (224, 224), color='gray')  # Fallback

        img_tensor = self.preprocess(img)  # Range [0, 1]
        return img_tensor, text


class LocalFlickrDataset(Dataset):
    """
    针对 Kaggle 下载的 Flickr30k/8k 数据集设计的加载器。
    读取本地 CSV/TXT 标注文件，并加载对应的图片。
    """

    def __init__(self, images_dir, ann_file, img_size=224, max_samples=None, delimiter='|', text_per_sample=5,
                 whitelist_path=None):
        """
        Args:
            whitelist_path (str): 包含图片ID的txt文件路径，用于指定加载特定的图片 (例如 'test.txt')
        """
        self.images_dir = images_dir
        self.img_size = img_size
        self.text_per_sample = text_per_sample

        # --- 1. 解析标注文件 ---
        print(f"[Dataset] Loading annotations from {ann_file}...")
        try:
            self.df = pd.read_csv(ann_file, sep=delimiter, on_bad_lines='skip', engine='python')
            self.df.columns = [c.strip() for c in self.df.columns]

            img_col = next((c for c in self.df.columns if 'image_name' == c.lower()), None)
            txt_col = next((c for c in self.df.columns if 'comment' == c.lower() or 'caption' in c.lower()), None)

            if not img_col or not txt_col:
                raise ValueError(f"无法识别列名。检测到的列名: {self.df.columns}")

            # === 新增：白名单过滤逻辑 ===
            if whitelist_path and os.path.exists(whitelist_path):
                print(f"[Dataset] Using whitelist from: {whitelist_path}")
                with open(whitelist_path, 'r') as f:
                    # 读取ID并去除空白符
                    valid_ids = set(line.strip() for line in f if line.strip())

                # 过滤函数：检查文件名（去除后缀）是否在 valid_ids 中
                # 假设 csv 中的 img_col 是 "1000092795.jpg"，而 test.txt 是 "1000092795"
                def is_in_whitelist(fname):
                    fname_str = str(fname)
                    base_name = os.path.splitext(fname_str)[0]  # 去除 .jpg
                    return base_name in valid_ids

                # 应用过滤
                original_len = len(self.df)
                self.df = self.df[self.df[img_col].apply(is_in_whitelist)]
                print(f"[Dataset] Filtered {original_len} -> {len(self.df)} rows based on whitelist.")
            # ==========================

            self.data = self.df[[img_col, txt_col]].values.tolist()

        except Exception as e:
            print(f"[Error] 解析标注文件失败: {e}")
            self.data = []

        # --- 2. 截取子集---
        if max_samples is not None:
            limit_count = max_samples * text_per_sample
            if len(self.data) > limit_count:
                self.data = self.data[:limit_count]
                print(f"[Dataset] Debug mode: limit to {max_samples} samples.")

        if whitelist_path is not None:
            print(f"[Dataset] Loaded whitelist subset. Final count: {len(self.data)}")

        # --- 3. 预处理 ---
        self.preprocess = Compose([
            Resize(img_size, interpolation=Image.BICUBIC),
            CenterCrop(img_size),
            ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, text = self.data[idx]
        img_path = os.path.join(self.images_dir, str(img_name).strip())
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (self.img_size, self.img_size), color='gray')

        image_tensor = self.preprocess(image)
        text = str(text).strip()
        return image_tensor, text, str(img_name)

# 辅助函数：由于攻击时需要手动 Normalize
class DifferentiableNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, img):
        # 支持梯度反向传播的 Normalize
        return (img - self.mean) / self.std
