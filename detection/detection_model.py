import torch
import torchvision
import pytorch_lightning as pl
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn, ssd300_vgg16
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class ObjectDetectionModel(pl.LightningModule):
    def __init__(self, model_type: str, num_classes: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        self.num_classes = num_classes
        self.lr = lr

        if model_type == "mask_rcnn":
            self.model = maskrcnn_resnet50_fpn(pretrained=True)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        elif model_type == "ssd300":
            self.model = ssd300_vgg16(pretrained=True)
            in_features = self.model.head.classification_head.num_classes
            self.model.head.classification_head = SSDClassificationHead(in_features, num_classes)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.metric = MeanAveragePrecision()

    def forward(self, images, targets=None):
        return self.model(images, targets) if self.training else self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        self.metric.update(outputs, targets)

    def validation_epoch_end(self, outputs):
        mAP = self.metric.compute()
        self.log("val_mAP", mAP["map"])
        self.metric.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)