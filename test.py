import os

import torchvision
import torch
from PIL import Image
import time
from model_AlexNet import AlexNet
from model_vgg16 import Vgg16

# 根据保存方式加载

# 加载方式1  网络与权重 一同保存，才可用此方式加载
# model = torch.load("AlexNet_104.pth", map_location=torch.device('cpu'))

# 加载方式2 只保存权重，才可用此方式加载
# 先加载 网络
# model = AlexNet()
model = Vgg16()

# 使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = model.to(device=device, dtype=torch.float32)


# 再加载 权重
# strict 权重与层数是否完整符合
model.load_state_dict(torch.load("vgg16_0_0.60.pth"), strict=True)

# 指定验证的图像路径
path = "./data/val/Cat"
# 验证的类别名称
# 定义类别对应字典
dist = {0: "猫", 1: "狗"}
l = "猫"

# ------------------------------------------------------------------------
# 注意更改缩放图像大小、维度转换时的图像大小
imgs = os.listdir(path)
len_imgs = len(imgs)
print(len_imgs)
# 总耗时
mean = 0
# 正确率
acc = 0

for i in imgs:
    # 读取图像
    img = Image.open(path + "/" + i)

    # 缩放、格式、归一化
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                                ])
    image = transform(img)
    # 注意维度转换，单张图片
    image1 = torch.reshape(image.to(device=device, dtype=torch.float32), (1, 3, 224, 224))

    a = time.time()
    # 测试开关
    model.eval()
    # 节约性能
    with torch.no_grad():
        output = model(image1)
        # print(output)
        print(output.cuda().cpu())
        # exit()
    # 转numpy格式,列表内取第一个
    a1 = dist[output.cuda().cpu().argmax(1).numpy()[0]]
    if a1 == l:
        acc += 1
    # print(a1, end="    ")
    mean += time.time() - a
    # img.show()

time_mean = mean / len_imgs
print("识别{}张图片，总耗时{}".format(len_imgs, mean))
print("平均耗时：{}".format(time_mean))
print("正确率：{}".format(acc / len_imgs))
