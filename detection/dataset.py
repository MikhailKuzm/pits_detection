import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PotholeDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        label_path = image_path.replace(".jpg", ".txt")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        with open(label_path, "r") as f:
            lines = f.readlines()

        boxes = []
        labels = []
        for line in lines:
            label, x, y, w, h = map(float, line.split())
            x_min = (x - w / 2) * image.shape[2]
            y_min = (y - h / 2) * image.shape[1]
            x_max = (x + w / 2) * image.shape[2]
            y_max = (y + h / 2) * image.shape[1]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(int(label) + 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return image, target


def get_dataloaders(root_dir, batch_size):
    dataset = PotholeDataset(root_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, val_loader