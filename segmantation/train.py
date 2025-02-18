import torch
 
import hydra
from hydra.utils import instantiate
import logging
import os
#os.chdir('segmantation') 
#log = logging.getLogger(__name__)
import numpy as np
import cv2
from dataset.pytorch_dataset import collate_fn
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(version_base=None, config_path='configs', config_name='pytorch_train')
def train(cfg): 
    print(cfg)
    model = instantiate(cfg['model']) 
    # Инициализируем дата-лоадеры
    train_loader = instantiate(cfg['data']['train_dataloader'], collate_fn = collate_fn)
    val_loader = instantiate(cfg['data']['val_dataloader'], collate_fn = collate_fn) 
    # Инициализируем trainer
    # --- Добавляем сохранение лучших чекпоинтов ---
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mAP",  # Мониторим метрику val_mAP
        mode="max",  # Чем больше, тем лучше
        save_top_k=3,  # Сохраняем только 3 лучших модели
        dirpath=f"{cfg.trainer.log_dir}/checkpoints",  # Папка для чекпоинтов
        filename="{epoch:02d}-{val_mAP:.4f}",  # Название файлов чекпоинтов
        save_weights_only=True  # Сохраняем только веса модели
    )
    trainer = instantiate(cfg['trainer'], callbacks=checkpoint_callback)#[instantiate(cfg.callbacks[clb_name]) for clb_name in cfg.callbacks]) 
    # Запуск обучения с передачей даталоадеров
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    train()
     