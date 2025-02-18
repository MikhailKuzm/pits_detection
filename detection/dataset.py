import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import to_pil_image 

class PotholeDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.to_tensor = T.ToTensor()
        # Получаем список всех изображений (фильтруем по .jpg)
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Загружаем изображение
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB').resize((576,576))
        orig_width, orig_height = image.size  # (W, H) в PIL
        image = self.to_tensor(image)
        

        # Загружаем соответствующие метки (bounding boxes)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))
        boxes = []
        labels = []

        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])  # Первый элемент — ID класса
                x_center, y_center, width, height = map(float, parts[1:])  # YOLO формат

                #x_min, y_min, x_max, y_max = map(float, parts[1:])
                # Преобразуем YOLO-формат в [x_min, y_min, x_max, y_max]
                x_min = (x_center - width / 2) * orig_width
                y_min = (y_center - height / 2) * orig_height
                x_max = (x_center + width / 2) * orig_width
                y_max = (y_center + height / 2) * orig_height

                boxes.append([x_min, y_min, x_max, y_max])
                #boxes.append([x_center, y_center, width, height])
                labels.append(class_id) 

        # Преобразуем в тензоры
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Создаём target
        target = {"boxes": boxes, "labels": labels}
        return image, target

def get_dataloaders(root_dir, batch_size=8):
    # Пути к данным
    train_img_dir = os.path.join(root_dir, "train", "images")
    train_lbl_dir = os.path.join(root_dir, "train", "labels")
    valid_img_dir = os.path.join(root_dir, "valid", "images")
    valid_lbl_dir = os.path.join(root_dir, "valid", "labels")

    # Создаём датасеты
    train_dataset = PotholeDataset(train_img_dir, train_lbl_dir)
    valid_dataset = PotholeDataset(valid_img_dir, valid_lbl_dir)

    # Создаём DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=64, shuffle=True,  collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=64, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, valid_loader


#import os
#os.chdir('detection')
#train_loader, valid_loader = get_dataloaders(root_dir = 'data') 
#batch = next(iter(train_loader))
#batch[1][0]['boxes']

#for item in batch[1]:
 #   ones_column = torch.ones((item['boxes'].shape[0], 1))
  #  item['boxes'] = torch.cat(( item['boxes'], ones_column), dim=1) 