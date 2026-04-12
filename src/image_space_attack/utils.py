from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

totensor = T.ToTensor()
topil = T.ToPILImage()
 
def recover_image(image, init_image, mask, background=False): # 将图像、初始图像和掩码转换为张量，并根据掩码将它们组合成最终图像
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background: # 用生成图像修改背景
        result = mask * init_image + (1 - mask) * image
    else: # 用生成图像修改掩码区域（需要生成nsfw内容的地方）
        result = mask * image + (1 - mask) * init_image
    return topil(result)

def preprocess(image): # 预处理
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32 map的作用是对w和h分别进行lambda定义的函数处理
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2) # 将图像从HWC格式转换为NCHW格式，并添加一个批次维度（None表示在第0维添加一个新的维度）
    image = torch.from_numpy(image) # 将numpy数组转换为torch张量（拆分了步骤所以要这样转换不用totensor）
    return 2.0 * image - 1.0

def prepare_mask_and_masked_image(image, mask): # 将PIL图像和掩码转换为张量，并根据掩码创建一个被遮挡的图像(加上标准化)
    image = np.array(image.convert("RGB")) # 将图像转换为RGB格式的numpy数组
    image = image[None].transpose(0, 3, 1, 2) # 将图像从HWC格式转换为NCHW格式，并添加一个批次维度（None表示在第0维添加一个新的维度）
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0 # 将图像像素值归一化到 -1到1 之间

    mask = np.array(mask.convert("L")) # 将掩码转换为灰度图像的numpy数组(单通道)
    mask = mask.astype(np.float32) / 255.0 # 将掩码转换为灰度图像，并将像素值归一化到 0到1 之间
    mask = mask[None, None] # 将掩码从HW格式转换为NCHW格式，并添加一个批次维度和通道维度（None表示在第0维和第1维添加新的维度）
    mask[mask < 0.5] = 0 # 将掩码中小于0.5(偏暗)的像素值设置为0，大于等于0.5（偏亮）的像素值设置为1，形成二值掩码
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask) # 将掩码转换为torch张量

    masked_image = image * (mask < 0.5) # 取反？

    return mask, masked_image

def prepare_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image[0]
 
