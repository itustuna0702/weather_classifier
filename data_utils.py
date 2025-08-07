import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split

def get_dataloaders(data_dir, batch_size, img_size, train_ratio, val_ratio):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_dir)
    targets = np.array(dataset.targets)
    indices = np.arange(len(targets))

    train_idx, temp_idx = train_test_split(indices, stratify=targets, test_size=1-train_ratio, random_state=42)
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(temp_idx, stratify=targets[temp_idx], test_size=1 - val_size, random_state=42)

    train_ds = Subset(datasets.ImageFolder(data_dir, transform=transform_train), train_idx)
    val_ds = Subset(datasets.ImageFolder(data_dir, transform=transform_eval), val_idx)
    test_ds = Subset(datasets.ImageFolder(data_dir, transform=transform_eval), test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader