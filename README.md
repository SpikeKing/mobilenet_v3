# MobileNet v3

[Paper](https://arxiv.org/abs/1905.02244): https://arxiv.org/abs/1905.02244

测试文件: test_of_mn3.py

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