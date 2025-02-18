import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np 
import os
import cv2 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import albumentations as A


def collate_fn(batch):
    ims, masks = zip(*batch)  # Распаковываем изображения и маски из списка кортежей
    #tfms = transforms.Compose([
    #                        transforms.ToTensor() 
    #                    ]) 
    #ims = torch.cat([tfms(im.copy())[None] for im in ims]).float()
    #ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long()
    #return ims, ce_masks
    ims = torch.stack([torch.tensor(im/255, dtype=torch.float32) for im in ims])  # Объединяем изображения в батч
    masks = torch.stack([torch.tensor(mask, dtype=torch.long) for mask in masks])  # Объединяем маски в батч
    
    return ims, masks


def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.02, p=1, border_mode=0
        ),
        #A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
        A.RandomCrop(height=200, width=200, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)



class PytData(Dataset):
    def __init__(self, image_path, mask_path, size, augmentation = None):
        self.image_path = image_path #путь к изображениям 
        self.mask_path = mask_path
        self.size = size
        self.items = os.listdir(image_path) #список всех изображений
        self.tfms = transforms.Compose([
                            transforms.ToTensor() 
                        ]) #преобразование изображения в тензор и номрализация
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.items)  
    
    def __getitem__(self, ix):
        image = cv2.imread(f'{self.image_path}/{self.items[ix]}', 1) #загружаем изображение 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #трансформируем BGR to RGB
        image = cv2.resize(image, (self.size,self.size)) 
        
        # Загружаем маску (она должна быть одноканальной)
        mask_path = os.path.join(self.mask_path, self.items[ix])#.replace(".jpg", ".png"))  # Убедимся, что расширение правильное
        mask_path = mask_path[:-4] + '_mask' + '.png' 
        mask = cv2.imread(mask_path, 0)  # Загружаем в grayscale (0 - одноканальная) 
        mask = cv2.resize(mask, (self.size, self.size))  # Приводим маску к единому размеру
        mask = np.expand_dims(mask, axis=-1)  # Добавляем канал (H, W, 1) 
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask  = sample["image"], sample["mask"]
            #mask = np.expand_dims(mask, axis=-1)  # Вернём обратно (H, W, 1)
        mask = cv2.resize(mask, (self.size,self.size))
        mask = np.expand_dims(mask, axis=-1) 
        image = cv2.resize(image, (self.size,self.size)) 
        return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)
         