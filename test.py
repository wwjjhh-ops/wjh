import torch
import numpy as np
from torch import nn

from config import device, model_path

from dataset import get_dataloaders
from model import VGG16_BN


# =====================================
# 1. 获取测试数据
# =====================================

train_loader, val_loader, test_loader = get_dataloaders()


# =====================================
# 2. 创建模型
# =====================================

vgg = VGG16_BN()

vgg = vgg.to(device)


# =====================================
# 3. 加载最佳模型
# =====================================

vgg.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

print("已加载最佳模型")


# =====================================
# 4. 损失函数
# =====================================

loss_fn = nn.CrossEntropyLoss()


# =====================================
# 5. 混淆矩阵
# =====================================

confusion_matrix = np.zeros(
    (2, 2),
    dtype=int
)


# =====================================
# 6. 测试
# =====================================

total_test_loss = 0

total_test_correct = 0

total_test_samples = 0


vgg.eval()


with torch.no_grad():

    for imgs, labels in test_loader:

        imgs = imgs.to(device)

        labels = labels.to(device)


        outputs = vgg(imgs)

        loss = loss_fn(
            outputs,
            labels
        )


        total_test_loss += loss.item()


        predictions = outputs.argmax(1)


        total_test_correct += (
            predictions == labels
        ).sum().item()


        total_test_samples += labels.size(0)


        # 统计混淆矩阵
        for true_label, pred_label in zip(
                labels,
                predictions):

            confusion_matrix[
                true_label.item(),
                pred_label.item()
            ] += 1


# =====================================
# 7. 输出结果
# =====================================

test_loss = (
    total_test_loss / len(test_loader)
)

test_accuracy = (
    total_test_correct /
    total_test_samples
)


print("\n============================")

print(
    "测试集损失：{:.4f}"
    .format(test_loss)
)

print(
    "测试集准确率：{:.3f}%"
    .format(test_accuracy * 100)
)

print("\n混淆矩阵：")

print(confusion_matrix)

print("============================")