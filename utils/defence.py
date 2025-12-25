import io
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import ToTensor
from concurrent.futures import ThreadPoolExecutor

def jpeg_compress_defense(image_tensor, quality=75):
    """
    模拟 JPEG 压缩防御
    Args:
        image_tensor: (C, H, W) 范围 [0, 1]
        quality: 压缩质量 1-100 (越低压缩越狠)
    Returns:
        defended_tensor: 压缩后再解压的图像
    """
    # 1. Tensor -> PIL
    # image_tensor 是 GPU 上的 float，先转成 CPU byte
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    img_pil = Image.fromarray(img_np)

    # 2. 内存中进行 JPEG 压缩与解压
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    img_defended = Image.open(buffer)

    # 3. PIL -> Tensor
    return ToTensor()(img_defended).to(image_tensor.device)

# --- 新增优化函数 ---
def _compress_one(args):
    """辅助函数：单张图片压缩，用于线程池"""
    img_np, quality = args
    img_pil = Image.fromarray(img_np)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    # 必须重新加载，否则是 lazy load
    return ToTensor()(Image.open(buffer))

def batch_jpeg_compress_defense(image_tensors, quality=75):
    """
    优化的并行 JPEG 防御
    Args:
        image_tensors: (B, C, H, W) GPU Tensor
    """
    device = image_tensors.device
    
    # 1. 批量转 CPU (比循环快)
    # (B, C, H, W) -> (B, H, W, C)
    images_np = (image_tensors.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
    
    # 2. 多线程并行压缩 (IO 密集型任务)
    # 3090 搭配的 CPU 通常核心数较多，设置为 8 或 16
    with ThreadPoolExecutor(max_workers=16) as executor:
        args = [(img, quality) for img in images_np]
        # list(executor.map) 会保持顺序
        defended_tensors = list(executor.map(_compress_one, args))

    # 3. 批量堆叠回 GPU
    return torch.stack(defended_tensors).to(device)