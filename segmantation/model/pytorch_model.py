import os 
import torch 
import pytorch_lightning as pl
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp  
from dataset.pytorch_dataset import PytData
from torch.utils.data import DataLoader 


class PytorchModel(pl.LightningModule):
    def __init__(self, architecture, encoder_name, in_channels, out_classes, 
                  num_workers, batch_size, max_epoch, learning_rate=2e-4):
        super().__init__()
        self.save_hyperparameters() 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epoch = max_epoch
        #self.trn_ds = PytData(image_path = image_path, mask_path = mask_path, train_ratio = train_ratio, 
        #                      mode = 'train', seed = 1, size = img_size)
        #self.val_ds = PytData(image_path = image_path, mask_path = mask_path, train_ratio = train_ratio,
        #                      mode = 'test', seed = 1,  size = img_size)
        
        self.model = smp.create_model(
            architecture,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes
        )
         
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage): 
        
        image, mask = batch
        logits_mask = self.forward(image)

        loss = self.loss_fn(logits_mask, mask)
        self.log(f'Loss_{stage}', loss, prog_bar=True, on_step=True, on_epoch=True)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        # Метрики IoU
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        
        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def shared_epoch_end(self, outputs, stage): 
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
 
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        self.log(f"{stage}_per_image_iou", per_image_iou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_dataset_iou", dataset_iou, prog_bar=True, on_epoch=True)


    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train") 
        self.training_step_outputs.clear() 

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid") 
        self.validation_step_outputs.clear() 

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        """Оптимизатор и scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epoch, eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}