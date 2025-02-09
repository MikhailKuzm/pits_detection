import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import os
sys.path.append('detection')
from dataset import get_dataloaders
from detection_model import ObjectDetectionModel
os.chdir('detection')
import torch
print(torch.cuda.is_available())

 
@hydra.main(version_base=None, config_path='configs', config_name="train")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    print(cfg)
    train_loader, val_loader = get_dataloaders(cfg.dataloader.root_dir, cfg.dataloader.batch_size)

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
        dirpath=f"{cfg.trainer.log_dir}\\checkpoints",  # Папка для чекпоинтов
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