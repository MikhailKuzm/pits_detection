import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import os
from ultralytics import YOLO
#sys.path.append('detection')
#os.chdir('detection') 
from dataset import get_dataloaders
from detection_model import ObjectDetectionModel
import torch
print(torch.cuda.is_available())
import yaml
 
 
@hydra.main(version_base=None, config_path='configs', config_name="train")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    print(cfg)
    train_loader, val_loader = get_dataloaders(cfg.dataloader.root_dir, cfg.dataloader.batch_size) 
    if cfg.model.name == 'yolo': 
        depth_values = [0.15, 0.25, 0.50, 0.75, 1.0, 1.5]
        width_values = [0.15, 0.25, 0.50, 0.75, 1.0, 1.5]
        # Путь к конфигурационному файлу YOLO
        yaml_path = "nets/YOLO/yolo11.yaml"
        # Читаем исходный yaml-файл
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        #for model_size in ["yolo11n.pt", "yolo11m.pt", "yolo11x.pt"]:
        for depth in depth_values:
            for width in width_values:
                # Изменяем параметры depth и width в `scales`
                config["scales"]['n'] = [depth, width, 2048]
                # Перезапись конфигурации в файл
                with open(yaml_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                model = YOLO(f'nets/YOLO/yolo11n.yaml')
                os.makedirs(f'logs/yolo11', exist_ok = True) 
                # Train the model
                model.train(data="nets/YOLO/data.yaml", epochs=cfg.trainer.epochs, imgsz=640, batch = cfg.dataloader.batch_size,
                                mosaic = 0, patience = 25, erasing = 0, shear = 0.1, name = f'extra13_[depth_{depth}-width_{width}]',
                                device = [0, 1,2,3,4,5, 6, 7], project = f'logs/yolo11')
        return        
    
    model = ObjectDetectionModel(
        model_type=cfg.model.name,
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
    )

    # --- Добавляем сохранение лучших чекпоинтов ---
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mAP",  # Мониторим метрику val_mAP
        mode="max",  # Чем больше, тем лучше
        save_top_k=3,  # Сохраняем только 3 лучших модели
        dirpath=f"{cfg.trainer.log_dir}/checkpoints",  # Папка для чекпоинтов
        filename="{epoch:02d}-{val_mAP:.4f}",  # Название файлов чекпоинтов
        save_weights_only=True  # Сохраняем только веса модели
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_steps,
        default_root_dir=cfg.trainer.log_dir,
        callbacks=[checkpoint_callback]  # Добавляем callback
    )

    trainer.fit(model, train_loader, val_loader)



if __name__ == "__main__":
    train()


