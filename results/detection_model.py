import torch 
import pytorch_lightning as pl
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 
from torchmetrics.detection.mean_ap import MeanAveragePrecision 


class ObjectDetectionModel(pl.LightningModule):
    def __init__(self, model_type: str, num_classes: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        self.num_classes = num_classes
        self.lr = lr
        #self.train_loss = MeanMetric() 
        self.model_type = model_type

        if model_type == "mask_rcnn":
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)#.to('cpu')
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        elif model_type == "ssd300":
            #self.model = ssd300_vgg16(pretrained=True)
            self.model = ssd300_vgg16(weights=True)
            #in_features = self.model.head.classification_head.num_classes
            #self.model.head.classification_head = SSDClassificationHead(in_features, num_classes)  
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.metric_train = MeanAveragePrecision()
        self.metric_val = MeanAveragePrecision()
        #self.training_step_losses = []
        #self.validation_step_losses = []
 

    def forward(self, images, targets=None):
        if targets:
            return self.model(images, targets)
        return self.model(images)

    def training_step(self, batch):
        images, targets = batch  
        # Прямой проход
        loss_dict = self.model(images, targets)  # Получаем потери 
        total_loss = sum(loss for loss in loss_dict.values())  # Суммируем все потери
        #self.training_step_losses.append(total_loss)

        # 🔄 Обновляем метрику
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)  # Предсказания для метрики 
        self.metric_train.update(outputs, targets) 

        # 🔄 Обновляем средний лосс
        #self.train_loss.update(total_loss)

        self.model.train()
        # 📝 Логируем лосс и метрики
        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True) 

        return total_loss

    def validation_step(self, batch):
        images, targets = batch 
        # Прямой проход (без градиентов)
        with torch.no_grad():
            # Получаем предсказания
            outputs = self.model(images)
        self.model.train()
        loss_dict = self.model(images, targets)  # Получаем потери 
         
        #print("AFTWR")
        #print(loss_dict)  
        total_loss = sum(loss for loss in loss_dict.values())  # Суммируем все потери в валидации
        self.model.eval()
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True) 
        # 🔄 Обновляем метрику (используем предсказания `outputs` и реальные метки `targets`)
        self.metric_val.update(outputs, targets)  
        
        return total_loss

    def on_train_epoch_end(self):
        mAP = self.metric_train.compute()
        #epoch_loss = self.train_loss.compute()
        self.log("train_mAP", mAP["map"]) 
        self.metric_train.reset()

    def on_validation_epoch_end(self):
        mAP = self.metric_val.compute()
        self.log("val_mAP", mAP["map"])
        self.metric_val.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
 