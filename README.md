# MobileNet v3

[Paper](https://arxiv.org/abs/1905.02244): https://arxiv.org/abs/1905.02244

测试文件: test_of_mn3.py

``` python
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
```

输出：

```text
[Info] 原始图片尺寸: (2000, 1250)
[Info] 变换之后的图像: torch.Size([3, 224, 224])
[Info] 增加1维: torch.Size([1, 3, 224, 224])
num classes:  100
[Info] 模型准确率: Epoch 37, Top1 69.63999938964844, Top5 90.98999786376953
[Info] 输出维度: torch.Size([100])
--------------------
[Info] 输出值: 33.843666076660156, 类别: woman
[Info] 输出值: 29.504314422607422, 类别: girl
[Info] 输出值: 24.901851654052734, 类别: boy
[Info] 输出值: 24.81878662109375, 类别: man
[Info] 输出值: 19.06907081604004, 类别: baby
```

[参考](https://github.com/leaderj1001/MobileNetV3-Pytorch)