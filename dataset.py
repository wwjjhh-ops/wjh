import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import root_dir, batch_size, seed


# =====================================
# 1. 检查图片是否损坏
# =====================================

def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False


# =====================================
# 2. 收集所有图片路径和标签
# =====================================

def get_images_and_labels():

    images = []
    labels = []

    # 猫：0
    cat_dir = os.path.join(root_dir, "Cat")

    for image_name in os.listdir(cat_dir):

        if not image_name.lower().endswith(
                (".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(cat_dir, image_name)

        if is_valid_image(image_path):
            images.append(image_path)
            labels.append(0)
        else:
            print("跳过损坏图片：", image_path)

    # 狗：1
    dog_dir = os.path.join(root_dir, "Dog")

    for image_name in os.listdir(dog_dir):

        if not image_name.lower().endswith(
                (".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(dog_dir, image_name)

        if is_valid_image(image_path):
            images.append(image_path)
            labels.append(1)
        else:
            print("跳过损坏图片：", image_path)

    print("总图片数量：", len(images))

    return images, labels


# =====================================
# 3. 划分数据集
# =====================================

def split_dataset(images, labels):

    total_size = len(images)

    generator = torch.Generator().manual_seed(seed)

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

    train_images = [images[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]

    val_images = [images[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    test_images = [images[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    return (
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
        test_labels
    )


# =====================================
# 4. 数据增强
# =====================================

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 随机旋转
    transforms.RandomRotation(10),
    transforms.ToTensor(),
transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
        )
])


# 验证集不进行随机增强
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# 测试集不进行随机增强
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# =====================================
# 5. 自定义 Dataset
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


# =====================================
# 6. 创建 DataLoader
# =====================================

def get_dataloaders():

    images, labels = get_images_and_labels()

    (
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
        test_labels
    ) = split_dataset(images, labels)

    train_dataset = Cat_DogDataset(
        train_images,
        train_labels,
        transform=train_transform
    )

    val_dataset = Cat_DogDataset(
        val_images,
        val_labels,
        transform=val_transform
    )

    test_dataset = Cat_DogDataset(
        test_images,
        test_labels,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    print("总数据：", len(images))
    print("训练集：", len(train_dataset))
    print("验证集：", len(val_dataset))
    print("测试集：", len(test_dataset))

    return train_loader, val_loader, test_loader
