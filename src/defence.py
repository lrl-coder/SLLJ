import io
from PIL import Image
from torchvision.transforms import ToTensor


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
