import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import cfg

class MyPipeline(pl.LightningModule):
    def __init__(self, learning_rate=None):
        super(MyPipeline, self).__init__()
        # Use config learning rate if not explicitly provided
        self.learning_rate = learning_rate if learning_rate is not None else cfg.LEARNING_RATE
        self.save_hyperparameters()
        
        # TODO: Define your model architecture here
        # Example:
        # self.encoder = ...
        # self.decoder = ...
        
    def forward(self, x):
        # TODO: Define the forward pass
        # Example: return self.decoder(self.encoder(x))
        pass

    def training_step(self, batch, batch_idx):
        # TODO: Implement the training step
        # x, y = batch
        # y_hat = self(x)
        # loss = F.mse_loss(y_hat, y)
        
        # Placeholder loss
        loss = torch.tensor(0.0, requires_grad=True)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: Implement the validation step
        # x, y = batch
        # y_hat = self(x)
        # val_loss = F.mse_loss(y_hat, y)
        
        # Placeholder loss
        val_loss = torch.tensor(0.0)
        
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss
        
    def test_step(self, batch, batch_idx):
        # TODO: Implement the test step
        # x, y = batch
        # y_hat = self(x)
        # test_loss = F.mse_loss(y_hat, y)
        
        # Placeholder loss
        test_loss = torch.tensor(0.0)
        
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self):
        # TODO: Define the optimizer and optionally learning rate schedulers
        # Example:
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # return optimizer
        
        # Returning a dummy optimizer so PyTorch Lightning doesn't crash if instantiated directly
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(1))], lr=self.learning_rate)
        return optimizer