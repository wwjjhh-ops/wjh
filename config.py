import torch
# =========================
# 设备
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 数据集路径
# =========================
root_dir = "/disk/WangJunHao/CatDogdataset/CatDog_dataset/PetImages"

# =========================
# 训练参数
# =========================
batch_size = 64
learning_rate = 0.001
epochs = 50

# =========================
# 早停
# =========================
patience = 5

# =========================
# 模型保存路径
# =========================
model_path = "vgg16_bn_best.pth"

# =========================
# 随机种子
# =========================
seed = 42
