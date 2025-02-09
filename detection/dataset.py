import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms

class PotholeDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.to_tensor = transforms.ToTensor()
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, valid_loader


























######################
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_bounding_boxes(images, targets, save_dir="tmp"):
    """
    Отрисовывает bounding boxes на изображениях и сохраняет результат, используя PIL.
    
    :param images: Список тензоров изображений (batch)
    :param targets: Список словарей с аннотациями (bounding boxes и labels)
    :param save_dir: Папка для сохранения изображений
    """
    os.makedirs(save_dir, exist_ok=True)  # ✅ Создаём папку, если её нет

    for i, (image, target) in enumerate(zip(images, targets)):
        # ✅ Преобразуем PyTorch Tensor в NumPy (если это не NumPy)
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()

        # ✅ Если изображение в формате `[C, H, W]`, меняем оси на `[H, W, C]`
        if image.shape[0] == 3:  # Каналы в первом измерении (C, H, W)
            image = np.transpose(image, (1, 2, 0))

        # ✅ Приводим значения в диапазон [0, 255] и делаем `uint8`
        image = (image * 255).astype(np.uint8)

        # ✅ Создаём объект изображения в PIL
        pil_image = Image.fromarray(image)

        # ✅ Создаём объект для рисования
        draw = ImageDraw.Draw(pil_image)

        # ✅ Получаем bounding boxes и метки
        boxes = target["boxes"].cpu().numpy()
        labels = target["labels"].cpu().numpy()

        # ✅ Отрисовываем bounding boxes
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            #x_center, y_center, width, height = box
            #x_min = (x_center - width / 2) * image.shape[1]
            #y_min = (y_center - height / 2) * image.shape[0]
            #x_max = (x_center + width / 2) * image.shape[1]
            #y_max = (y_center + height / 2) * image.shape[0]
            print([x_min, y_min, x_max, y_max])
            draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=5)
            draw.text((x_min, y_min - 10), f"Class {label}", fill="green")

        # ✅ Сохраняем изображение
        save_path = os.path.join(save_dir, f"annotated_{i}.jpg")
        pil_image.save(save_path)
        print(f"✅ Изображение сохранено: {save_path}")


# 📌 Пример использования
train_loader, valid_loader = get_dataloaders(root_dir="D:\\study\\pits_detection\\detection\\data")

# Получаем batch
tmp = next(iter(train_loader))
images, targets = tmp[0], tmp[1]

# Отрисовываем боксы и сохраняем изображения
draw_bounding_boxes(images, targets)


# Получаем batch
tmp = next(iter(train_loader))
images, targets = tmp[0], tmp[1]

img = images[0]
from PIL import Image
from torchvision.transforms.functional import to_pil_image
pil_img = to_pil_image(img)
pil_img.save(r'detection\tmp\tmp.jpg')



# Отрисовываем боксы и сохраняем изображения
draw_bounding_boxes(images, targets)

