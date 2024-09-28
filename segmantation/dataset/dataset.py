import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import os
import cv2
from sklearn.model_selection import train_test_split
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SegData(Dataset):
    def __init__(self, image_path, mask_path, train_ratio = 0.9,  mode = 'train', seed = 1):
        self.image_path = image_path #путь к изображениям 
        self.mask_path = mask_path
        files = np.array(os.listdir(image_path)) #список всех изображений
        
        np.random.seed(seed)
        train_indx = random.sample(range(len(files)), int(len(files)*train_ratio))  
        if mode == 'train':
            self.items = files[train_indx] #список имён изображений для train
        elif mode == 'test': 
            self.items = files[~np.isin(np.arange(len(files)), train_indx)]  #список имён изображений для test
         
        self.tfms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
                        ]) #преобразование изображения в тензор и номрализация
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, ix):
        image = cv2.imread(f'{self.image_path}/{self.items[ix]}', 1) #загружаем изображение
        
        image = cv2.resize(image, (224,224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #трансформируем BGR to RGB
       # image = np.transpose(image, (2, 0, 1)) # [channels, width, height] 
        
        mask = cv2.imread(f'{self.mask_path}/{self.items[ix][:-3]}png', 0) #загружаем маску 
        mask = cv2.resize(mask, (224,224))
        
        return image, mask
    
    #def choose(self): return self[randint(len(self))]
    
    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        ims = torch.cat([self.tfms(im.copy()/255)[None] for im in ims]).float()
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long()
        
        return ims, ce_masks