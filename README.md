说明文档：本项目用于学习图像分类的教学项目
数据集下载：

链接: https://pan.baidu.com/s/1vK2ITIjAPiZUQD7OfsEc8A  密码: 1k51

先安装环境 ----> 使用data_classify.py文件进行训练集与测试集分割 ----> 在进行训练即可


数据准备：当前数据存放 data_name 文件夹内

文件夹名就是类别名，n个类别就是n个文件夹

## 目录主要结构组成：

model_AlexNet.py   ---->  自己建的AlexNet模型（可选其他模型）

model_Vgg16.py      ---->  pytorch自带更改的模型（可选其他模型）

train.py					  ---->  用于训练模型

test.py					    ---->  用于测试模型


### 辅助文件：

data_classify.py   ----> 将 data_name内的类别分为训练集与测试集。

​										注意查看代码内容，包含argparse模块

清除单通道图像   -----> 数据清洗，处理异常图像

旧版数据加载	   -----> 用于学习图像  数据加载

## 安装环境：

pip install -r requirements.txt

pip install tensorboard

去pytorch官方下载：自己对应cuda版本的pytorch，如果没有GPU 下载cpu也可

