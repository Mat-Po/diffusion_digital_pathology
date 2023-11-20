import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.functional import accuracy
from torchvision.models import resnet50
from torch.optim.lr_scheduler import CosineAnnealingLR
from vit_pytorch import SimpleViT



class res50(LightningModule):
    def __init__(self, num_classes, learning_rate=2e-4,tmax = 20):

        super().__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.resnet_model = resnet50()
        linear_size = list(self.resnet_model.children())[-1].in_features
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)
        self.loss_function = nn.CrossEntropyLoss()
        self.tmax = tmax

    

    def forward(self, x):
        x = self.resnet_model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y_hat = torch.argmax(preds, dim=1)
        task_type = "multiclass" if self.num_classes > 2 else "binary"
        acc = accuracy(y_hat, y,task_type,num_classes=self.num_classes)
        self.log("Training_acc", acc, on_epoch=True, prog_bar=True,on_step=False)

        loss = self.loss_function(preds, y)
        self.log('Training_loss',loss.item(),prog_bar= True,on_step=True,on_epoch=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.resnet_model(x)
        loss = self.loss_function(preds, y) 
        y_hat = torch.argmax(preds, dim=1)
        task_type = "multiclass" if self.num_classes > 2 else "binary"
        acc = accuracy(y_hat, y,task_type,num_classes=self.num_classes)
        self.log("val_loss", loss.item(), prog_bar= False,on_epoch=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.resnet_model(x)
        loss = self.loss_function(preds, y) 
        y_hat = torch.argmax(preds, dim=1)
        task_type = "multiclass" if self.num_classes > 2 else "binary"
        acc = accuracy(y_hat, y,task_type,num_classes=self.num_classes)
        self.log("test_loss", loss.item(), prog_bar= False,on_epoch=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer,T_max=self.tmax)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": 'epoch',
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        }
    }



class VIT_model(LightningModule):
    def __init__(self,image_size,patch_size,num_classes,dim,depth,heads,mlp_dim,tmax = 20,learning_rate=2e-4):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim 
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()
        self.tmax = tmax
        self.vit_model = SimpleViT(image_size=  self.image_size,
                            patch_size = self.patch_size,
                            num_classes = self.num_classes,
                            dim = self.dim,
                            depth = self.depth,
                            heads = self.heads,
                            mlp_dim = self.mlp_dim)
        self.save_hyperparameters()
    def forward(self, x):
        out = self.vit_model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        y_hat = torch.argmax(preds, dim=-1)
        task_type = "multiclass" if self.num_classes > 2 else "binary"
        acc = accuracy(y_hat, y,task_type,num_classes=self.num_classes)
        self.log("Training_acc", acc, on_epoch=True, prog_bar=True,on_step=False)

        loss = self.loss_function(preds, y)
        self.log('Training_loss',loss.item(),prog_bar= True,on_step=True,on_epoch=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_function(preds, y) 
        y_hat = torch.argmax(preds, dim=-1)
        task_type = "multiclass" if self.num_classes > 2 else "binary"
        acc = accuracy(y_hat, y,task_type,num_classes=self.num_classes)
        self.log("val_loss", loss.item(), prog_bar= False,on_epoch=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_function(preds, y) 
        y_hat = torch.argmax(preds, dim=-1)
        task_type = "multiclass" if self.num_classes > 2 else "binary"
        acc = accuracy(y_hat, y,task_type,num_classes=self.num_classes)
        self.log("test_loss", loss.item(), prog_bar= False,on_epoch=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer,T_max=self.tmax)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": 'epoch',
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        }
    }
            