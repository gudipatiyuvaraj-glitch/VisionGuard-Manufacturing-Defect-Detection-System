"""
dataset.py - Data preparation and DataLoader pipeline with augmentations.
"""
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Standard ImageNet normalization values
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size: int = 224):
    """Returns training and validation/test transformation pipelines."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    return train_transform, eval_transform

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Builds DataLoaders for train, validation, and test splits."""
    train_transform, eval_transform = get_transforms(img_size)

    train_set = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    val_set = datasets.ImageFolder(root=f"{data_dir}/val", transform=eval_transform)
    test_set = datasets.ImageFolder(root=f"{data_dir}/test", transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_set.class_to_idx