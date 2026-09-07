import os
import torch
from PIL import Image
import torchvision.transforms as transforms

from model import VGG16_BN


# =====================================
# 1. 设置设备
# =====================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("当前设备：", device)


# =====================================
# 2. 图片预处理
# =====================================

# 注意：
# 预测时不能使用 train_transform
# 因为 train_transform 中有随机翻转和随机旋转。
#
# 预测自己的图片时，使用和验证集 / 测试集
# 相同的预处理方式。

transform = transforms.Compose([
    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# =====================================
# 3. 加载模型
# =====================================

model = VGG16_BN()

model_path = "vgg16_bn_best.pth"

if not os.path.exists(model_path):
    print("找不到模型文件：", model_path)
    print("请确认 vgg16_bn_best.pth 在项目根目录下。")
    exit()


# 加载训练好的模型参数
model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

# 将模型移动到 GPU / CPU
model = model.to(device)

# 开启推理模式
model.eval()

print("模型加载成功！")


# =====================================
# 4. 预测函数
# =====================================

def predict_image(image_path):

    # 检查图片是否存在
    if not os.path.exists(image_path):
        print("找不到图片：", image_path)
        return

    # 读取图片
    image = Image.open(image_path).convert("RGB")

    # 图片预处理
    image_tensor = transform(image)

    # 增加 Batch 维度
    #
    # 原来：
    # [3, 128, 128]
    #
    # 变成：
    # [1, 3, 128, 128]
    image_tensor = image_tensor.unsqueeze(0)

    # 移动到 GPU / CPU
    image_tensor = image_tensor.to(device)

    # 关闭梯度计算
    with torch.no_grad():

        # 模型预测
        output = model(image_tensor)

        # 转换成概率
        probabilities = torch.softmax(
            output,
            dim=1
        )

        # 获取概率最大的类别
        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    # 获取两个类别的概率
    cat_probability = probabilities[0][0].item()
    dog_probability = probabilities[0][1].item()

    # =====================================
    # 5. 输出结果
    # =====================================

    if predicted_class == 0:
        result = "Cat 🐱"
    else:
        result = "Dog 🐶"

    print()
    print("==============================")
    print("图片：", os.path.basename(image_path))
    print()
    print("预测结果：", result)
    print()
    print("Cat 概率：{:.2f}%".format(
        cat_probability * 100
    ))

    print("Dog 概率：{:.2f}%".format(
        dog_probability * 100
    ))

    print("==============================")


# =====================================
# 6. 输入图片路径
# =====================================

while True:

    print()
    image_path = input(
        "请输入图片路径（输入 q 退出）："
    )

    # 输入 q 退出
    if image_path.lower() == "q":
        print("程序结束。")
        break

    # 去掉可能存在的引号
    image_path = image_path.strip().strip('"')

    # 进行预测
    predict_image(image_path)

