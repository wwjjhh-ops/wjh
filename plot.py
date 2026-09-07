import torch
import matplotlib.pyplot as plt


# =====================================
# 1. 加载训练历史
# =====================================

history = torch.load(
    "training_history.pth"
)


train_losses = history["train_losses"]

train_accuracies = history["train_accuracies"]

val_losses = history["val_losses"]

val_accuracies = history["val_accuracies"]


epochs = range(
    1,
    len(train_losses) + 1
)


# =====================================
# 2. Loss 曲线
# =====================================

plt.figure()

plt.plot(
    epochs,
    train_losses,
    label="Train Loss"
)

plt.plot(
    epochs,
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.title(
    "Training and Validation Loss"
)


# =====================================
# 3. Accuracy 曲线
# =====================================

plt.figure()

plt.plot(
    epochs,
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    epochs,
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.title(
    "Training and Validation Accuracy"
)


plt.show()