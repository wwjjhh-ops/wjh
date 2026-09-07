import torch
from torch import nn

from config import (
    device,
    learning_rate,
    epochs,
    patience,
    model_path
)

from dataset import get_dataloaders
from model import VGG16_BN


# =====================================
# 1. 获取数据
# =====================================

train_loader, val_loader, test_loader = get_dataloaders()


# =====================================
# 2. 创建模型
# =====================================

vgg = VGG16_BN()

vgg = vgg.to(device)

print("当前设备：", device)


# =====================================
# 3. 损失函数
# =====================================

loss_fn = nn.CrossEntropyLoss()


# =====================================
# 4. 优化器
# =====================================

optimizer = torch.optim.SGD(
    vgg.parameters(),
    lr=learning_rate
)


# =====================================
# 5. 学习率调度器
# =====================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=2
)


# =====================================
# 6. 记录训练数据
# =====================================

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []


# =====================================
# 7. 早停
# =====================================

best_val_accuracy = 0.0

patience_counter = 0


# =====================================
# 8. 开始训练
# =====================================

for i in range(epochs):

    print(
        "\n第{}轮训练开始-----------------------"
        .format(i + 1)
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"当前学习率：{current_lr:.6f}"
    )


    # =================================
    # 训练
    # =================================

    vgg.train()

    total_train_loss = 0

    total_train_correct = 0

    total_train_samples = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        outputs = vgg(images)

        loss = loss_fn(outputs, labels)


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_train_loss += loss.item()

        total_train_correct += (
            outputs.argmax(1) == labels
        ).sum().item()

        total_train_samples += labels.size(0)


    train_loss = (
        total_train_loss / len(train_loader)
    )

    train_accuracy = (
        total_train_correct /
        total_train_samples
    )


    train_losses.append(train_loss)

    train_accuracies.append(train_accuracy)


    print(
        "训练集 Loss：{:.4f}"
        .format(train_loss)
    )

    print(
        "训练集准确率：{:.3f}%"
        .format(train_accuracy * 100)
    )


    # =================================
    # 验证
    # =================================

    vgg.eval()

    total_val_loss = 0

    total_val_correct = 0

    total_val_samples = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = vgg(images)

            loss = loss_fn(
                outputs,
                labels
            )


            total_val_loss += loss.item()

            total_val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

            total_val_samples += labels.size(0)


    val_loss = (
        total_val_loss / len(val_loader)
    )

    val_accuracy = (
        total_val_correct /
        total_val_samples
    )


    val_losses.append(val_loss)

    val_accuracies.append(val_accuracy)


    print(
        "验证集 Loss：{:.4f}"
        .format(val_loss)
    )

    print(
        "验证集准确率：{:.3f}%"
        .format(val_accuracy * 100)
    )


    # =================================
    # 学习率调整
    # =================================

    scheduler.step(val_loss)


    # =================================
    # 保存最佳模型
    # =================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            vgg.state_dict(),
            model_path
        )

        patience_counter = 0

        print("发现更好的模型，已保存！")

    else:

        patience_counter += 1

        print(
            "验证集没有提升，"
            "当前连续{}轮没有提升"
            .format(patience_counter)
        )


        # =================================
        # Early Stopping
        # =================================

        if patience_counter >= patience:

            print(
                "连续 {} 轮验证集没有提升，"
                "提前停止训练！"
                .format(patience)
            )

            break


# =====================================
# 9. 保存训练曲线数据
# =====================================

torch.save({

    "train_losses": train_losses,

    "train_accuracies": train_accuracies,

    "val_losses": val_losses,

    "val_accuracies": val_accuracies

}, "training_history.pth")


print("\n训练完成！")

print(
    "最佳验证集准确率：{:.3f}%"
    .format(best_val_accuracy * 100)
)