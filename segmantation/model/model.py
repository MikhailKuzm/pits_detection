
import pytorch_lightning as pl 
import torch
from dataset.dataset import SegData
from torch.utils.data import DataLoader 

class SegModel(pl.LightningModule):

    def __init__(self, image_path, mask_path, train_ratio,  num_workers, batch_size, net):
        super().__init__() 
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.trn_ds = SegData(image_path = image_path, mask_path = mask_path, train_ratio = train_ratio,  mode = 'train', seed = 1)
        self.val_ds = SegData(image_path = image_path, mask_path = mask_path, train_ratio = train_ratio,  mode = 'test', seed = 1)
        #self.train_dataloader = DataLoader(trn_ds, batch_size=4, shuffle=True, collate_fn=trn_ds.collate_fn)
       # self.test_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=val_ds.collate_fn)
        
        self.loss_fn = torch.nn.BCELoss()
        self.net = net
        
        
    def training_step(self, batch):
        return self.shared_step(batch, "train")            

    #def on_train_epoch_end(self, outputs):
    #    return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch):
        return self.shared_step(batch, "valid")

    #def on_validation_epoch_end(self, outputs):
    #    return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    #def on_test_epoch_end(self, outputs):
    #    return self.shared_epoch_end(outputs, "test")
 

    def shared_step(self, batch, stage):
        
        image, true_mask = batch
        logits_mask = self.net(image).squeeze(1)
        loss = self.loss_fn(logits_mask.float(), true_mask.float()) 
        acc = (logits_mask.round() == true_mask).sum().item()/(self.batch_size*224*224)
        
        self.log(f'Loss_{stage}', loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'Accuracy_{stage}', acc, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
        

    #def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        
        
    def train_dataloader(self):
        return DataLoader(self.trn_ds, batch_size=self.batch_size, num_workers = self.num_workers, shuffle=True, 
                          persistent_workers=True, collate_fn=self.trn_ds.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,  num_workers = self.num_workers, shuffle=False, 
                          persistent_workers=True, collate_fn=self.val_ds.collate_fn)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=0.001)
         

    

