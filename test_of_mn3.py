#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/5/11
"""
import os
import torch
from PIL import Image
from torchvision.transforms import transforms

from mn3_model import MobileNetV3
from root_dir import IMGS_DIR, MODELS_DIR

# CIFAR100的标签列表 100个
CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


def test_of_mn3():
    img_path = os.path.join(IMGS_DIR, 'woman.jpg')
    img_pil = Image.open(img_path)
    print('[Info] 原始图片尺寸: {}'.format(img_pil.size))

    # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR100的参数
    ])

    img_torch = trans(img_pil)  # 标准变换
    print("[Info] 变换之后的图像: {}".format(img_torch.shape))

    img_torch = torch.unsqueeze(img_torch, 0).to(torch.device("cpu"))
    print("[Info] 增加1维: {}".format(img_torch.shape))

    # 100维输出，cpu模式
    mode_type = 'LARGE'  # LARGE or SMALL
    model = MobileNetV3(model_mode=mode_type, num_classes=100, multiplier=1.0).to(torch.device("cpu"))
    model_pretrained = os.path.join(MODELS_DIR, 'mn3_model_{}_ckpt.t7'.format(mode_type))
    checkpoint = torch.load(model_pretrained, map_location='cpu')  # 读取模型的CPU版本
    model.load_state_dict(checkpoint['model'])  # 加载模型

    epoch = checkpoint['epoch']
    acc1 = checkpoint['best_acc1']
    acc5 = checkpoint['best_acc5']
    print('[Info] 模型准确率: Epoch {}, Top1 {}, Top5 {}'.format(epoch, acc1, acc5))

    # squeeze_model = models.squeezenet1_1(pretrained=True)

    model.eval()  # 转换为评估模式
    output = model(img_torch)[0]  # 预测图片
    print('[Info] 输出维度: {}'.format(output.shape))
    _, pred = output.topk(5, 0, True, True)  # Top5
    print('-' * 20)
    for x in pred.data.numpy():
        val = output[x]
        clz_name = CIFAR100_LABELS_LIST[x]
        print('[Info] 输出值: {}, 类别: {}'.format(val, clz_name))


def main():
    test_of_mn3()


if __name__ == '__main__':
    main()
