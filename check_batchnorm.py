import os
from PIL import Image
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# =====================================
# 1. 设备
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备：", device)


# =====================================
# 2. 数据路径
# =====================================

root_dir = "D:\\pycharmprojects\\learn_pytorch\\CatDog_dataset\\PetImages"


# =====================================
# 3. 检查图片
# =====================================

def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False


images = []
labels = []


# 猫 = 0
cat_dir = os.path.join(root_dir, "Cat")

for image_name in os.listdir(cat_dir):

    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(cat_dir, image_name)

    if is_valid_image(image_path):
        images.append(image_path)
        labels.append(0)


# 狗 = 1
dog_dir = os.path.join(root_dir, "Dog")

for image_name in os.listdir(dog_dir):

    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(dog_dir, image_name)

    if is_valid_image(image_path):
        images.append(image_path)
        labels.append(1)


print("总图片数量：", len(images))


# =====================================
# 4. 必须使用和训练时完全一样的划分
# =====================================

total_size = len(images)

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    total_size,
    generator=generator
)

train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)

train_indices = indices[:train_size]

val_indices = indices[
    train_size:train_size + val_size
]

test_indices = indices[
    train_size + val_size:
]


# =====================================
# 5. 获取验证集
# =====================================

val_images = [images[i] for i in val_indices]
val_labels = [labels[i] for i in val_indices]


print("验证集：", len(val_images))


# =====================================
# 6. 验证集预处理
# =====================================

val_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

])


# =====================================
# 7. Dataset
# =====================================

class Cat_DogDataset(Dataset):

    def __init__(self, images, labels, transform=None):

        self.images = images
        self.labels = labels
        self.transform = transform


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        image_path = self.images[index]

        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


val_dataset = Cat_DogDataset(
    val_images,
    val_labels,
    transform=val_transform
)


val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    pin_memory=True
)


# =====================================
# 8. VGG16_BN
# =====================================

class VGG16_BN(nn.Module):

    def __init__(self):

        super(VGG16_BN, self).__init__()

        self.model1 = nn.Sequential(

            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(2),
        )


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.model2 = nn.Sequential(

            nn.Linear(512, 256),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256, 2)
        )


    def forward(self, x):

        x = self.model1(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.model2(x)

        return x


# =====================================
# 9. 加载已经训练好的最佳模型
# =====================================

vgg = VGG16_BN().to(device)

vgg.load_state_dict(
    torch.load(
        "vgg16_bn_best.pth",
        map_location=device
    )
)

print("最佳模型加载成功！")


# =====================================
# 10. 正常验证
# =====================================

loss_fn = nn.CrossEntropyLoss()


def evaluate_normal():

    vgg.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = vgg(images)

            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    accuracy = total_correct / total_samples

    loss = total_loss / len(val_loader)

    return loss, accuracy


# =====================================
# 11. 只让 BatchNorm 使用 batch 统计量
# =====================================

def evaluate_batchnorm_batch_statistics():

    # 整个模型先进入 eval
    # 这样 Dropout 仍然关闭

    vgg.eval()


    # 找到所有 BatchNorm
    # 只把 BatchNorm 设置为 train

    for module in vgg.modules():

        if isinstance(module, nn.BatchNorm2d):

            module.train()


    total_correct = 0
    total_samples = 0
    total_loss = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = vgg(images)

            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)


    accuracy = total_correct / total_samples

    loss = total_loss / len(val_loader)

    return loss, accuracy


# =====================================
# 12. 两种方法进行比较
# =====================================

normal_loss, normal_accuracy = evaluate_normal()

bn_loss, bn_accuracy = evaluate_batchnorm_batch_statistics()


print()
print("====================================")
print("BatchNorm 诊断结果")
print("====================================")

print(
    "正常 eval()："
    f"Loss = {normal_loss:.4f}, "
    f"Accuracy = {normal_accuracy * 100:.3f}%"
)

print(
    "BatchNorm 使用当前 batch："
    f"Loss = {bn_loss:.4f}, "
    f"Accuracy = {bn_accuracy * 100:.3f}%"
)

print("====================================")


difference = (
    bn_accuracy - normal_accuracy
) * 100


print(
    f"准确率差异：{difference:+.3f}%"
)


# =====================================
# 验证集 Batch 诊断
# =====================================

print()
print("====================================")
print("验证集 Batch 诊断")
print("====================================")

vgg.eval()

batch_accuracies = []
batch_losses = []

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(val_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = vgg(images)

        loss = loss_fn(outputs, labels)

        predictions = outputs.argmax(1)

        correct = (predictions == labels).sum().item()
        total = labels.size(0)

        batch_accuracy = correct / total

        batch_accuracies.append(batch_accuracy)
        batch_losses.append(loss.item())

        print(
            f"Batch {batch_idx + 1:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {batch_accuracy * 100:.2f}%"
        )


# =====================================
# 最差的 10 个 Batch
# =====================================

print()
print("====================================")
print("最差的 10 个 Batch")
print("====================================")

worst_batches = sorted(
    enumerate(batch_accuracies),
    key=lambda x: x[1]
)[:10]

for batch_idx, accuracy in worst_batches:

    print(
        f"Batch {batch_idx + 1:3d} | "
        f"Accuracy: {accuracy * 100:.2f}% | "
        f"Loss: {batch_losses[batch_idx]:.4f}"
    )


# =====================================
# 最好的 10 个 Batch
# =====================================

print()
print("====================================")
print("最好的 10 个 Batch")
print("====================================")

best_batches = sorted(
    enumerate(batch_accuracies),
    key=lambda x: x[1],
    reverse=True
)[:10]

for batch_idx, accuracy in best_batches:

    print(
        f"Batch {batch_idx + 1:3d} | "
        f"Accuracy: {accuracy * 100:.2f}% | "
        f"Loss: {batch_losses[batch_idx]:.4f}"
    )


# =====================================
# 重复验证 3 次
# =====================================

print()
print("====================================")
print("验证集重复测试")
print("====================================")

for run in range(3):

    vgg.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = vgg(images)

            predictions = outputs.argmax(1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    accuracy = total_correct / total_samples

    print(
        f"第 {run + 1} 次验证："
        f"{accuracy * 100:.3f}%"
    )

