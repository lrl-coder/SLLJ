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

    def __init__(self, images_dir, ann_file, img_size=224, max_samples=None, delimiter='|', text_per_sample=5):
        """
        Args:
            images_dir (str): 存放图片的文件夹路径 (例如 'flickr30k_images/')
            ann_file (str): 标注文件路径 (例如 'results.csv')
            img_size (int): 图片缩放大小
            max_samples (int): 调试用，仅加载前 N 张图 (设为 None 则加载全部)
            delimiter (str): CSV 分隔符，Flickr30k 通常是 '|', Flickr8k 可能是 ','
        """
        self.images_dir = images_dir
        self.img_size = img_size
        self.text_per_sample = text_per_sample
        

        # --- 1. 解析标注文件 ---
        print(f"[Dataset] Loading annotations from {ann_file}...")
        try:
            # Kaggle Flickr30k 的 results.csv 有时会有表头，有时没有
            # 常见格式: image_name | comment_number | comment
            # 我们使用 pandas 读取，并在出错时跳过坏行 (on_bad_lines='skip')
            self.df = pd.read_csv(ann_file, sep=delimiter, on_bad_lines='skip', engine='python')

            # 清洗列名：有些数据集列名带有空格，如 ' comment'
            self.df.columns = [c.strip() for c in self.df.columns]

            # 确保找到对应的列
            # Flickr30k常见列名: 'image_name', 'comment'
            img_col = next((c for c in self.df.columns if 'image_name' == c.lower()), None)
            txt_col = next((c for c in self.df.columns if 'comment' == c.lower() or 'caption' in c.lower()), None)

            if not img_col or not txt_col:
                raise ValueError(f"无法识别列名。检测到的列名: {self.df.columns}")

            # （全量comment）
            self.data = self.df[[img_col, txt_col]].values.tolist()

        except Exception as e:
            print(f"[Error] 解析标注文件失败: {e}")
            # 如果 Pandas 失败，回退到一个空列表，防止程序崩溃
            self.data = []

        # --- 2. 截取子集 (调试用) ---
        if max_samples is not None:
            self.data = self.data[:max_samples * text_per_sample]
            print(f"[Dataset] Debug mode: limit to {max_samples} samples.")

        print(f"[Dataset] Loaded {len(self.data)} image-text pairs.")

        # --- 3. 预处理 ---
        self.preprocess = Compose([
            Resize(img_size, interpolation=Image.BICUBIC),
            CenterCrop(img_size),
            ToTensor()  # [0, 1]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取文件名和文本
        img_name, text = self.data[idx]

        # 构造完整路径
        img_path = os.path.join(self.images_dir, str(img_name).strip())

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (self.img_size, self.img_size), color='gray')

        image_tensor = self.preprocess(image)
        text = str(text).strip()

        # 多返回一个 img_name 用于后续 ID 匹配
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
