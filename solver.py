import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from audiodata import SpeechDataset


class SpeechSeparationModel(pl.LightningModule):

    def __init__(self, net, train_set: Dataset, val_set: Dataset):
        super(SpeechSeparationModel, self).__init__()
        self.net = net
        self.train_set = train_set
        self.val_set = val_set

    def forward(self, data):
        """TODO"""
        pass

    def training_step(self, batch, batch_nb):
        """TODO"""
        pass

    def validation_step(self, batch, batch_nb):
        """TODO"""
        pass

    def validation_epoch_end(self, outputs):
        """TODO"""
        pass

    def configure_optimizers(self):
        """TODO"""
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_set, batch_size=32)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_set, batch_size=32)


if __name__ == "__main__":
    root_dir = ""
    train_file = ""
    val_file = ""
    train_set = SpeechDataset(root_dir, train_file)
    val_set = SpeechDataset(root_dir, val_file)
    speech_separation = SpeechSeparationModel()

    trainer = pl.Trainer(gpus=1)
    trainer.fit(speech_separation)
