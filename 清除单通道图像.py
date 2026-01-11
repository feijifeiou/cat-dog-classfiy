import os
import numpy as np
from PIL import Image


# 功能：是清除 单通道、大于3通道、的图像

# 要求：文件目录下均为 图像
def remove_img(path):
    data_list = os.listdir(path)
    n = 0
    n1 = 0
    max_data = len(data_list)
    for i in data_list:
        path_img = path + "/" + i
        img = np.array(Image.open(path_img))
        print("\r进度{}---{}".format(n, max_data), end="")
        n += 1
        if len(img.shape) != 3 or img.shape[2] > 3:
            n1 += 1
            os.remove(path_img)
    print()
    print("{}--删除{}个单通道图像 或者 是大于3通道图像".format(path, n1))


def data_list(path):
    data = os.listdir(path)
    for i in data:
        path_list = path + "/" + i
        remove_img(path_list)
    print("运行完毕")


if __name__ == '__main__':
    path = "./data_name"
    data_list(path)
