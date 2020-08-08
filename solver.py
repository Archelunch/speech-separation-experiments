import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from audiodata import SpeechDataset
from model import ConvTasNet
from pit_criterion import cal_loss


class SpeechSeparationModel(pl.LightningModule):

    def __init__(self, net, train_set: Dataset, val_set: Dataset):
        super(SpeechSeparationModel, self).__init__()
        self.net = net
        self.train_set = train_set
        self.val_set = val_set

    def forward(self, data):
        return self.net.forward(data)

    def training_step(self, batch, batch_nb):
        x, _ = batch

        padded_mixture, mixture_lengths, padded_source = batch
        estimate_source = self.forward(padded_mixture)

        loss, max_snr, estimate_source, reorder_estimate_source = \
            cal_loss(padded_source, estimate_source, mixture_lengths)

        # ?
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(),
        #                                self.max_norm)

        # ?
        # total_loss += loss.item()

        return {
            'loss': loss.item(),
            'log': f'Batch index: {batch_nb}. Loss: {loss.item()}'
        }

    def validation_step(self, batch, batch_nb):
        x, _ = batch

        padded_mixture, mixture_lengths, padded_source = batch
        estimate_source = self.forward(padded_mixture)

        loss, max_snr, estimate_source, reorder_estimate_source = \
            cal_loss(padded_source, estimate_source, mixture_lengths)

        return {'val_loss': loss.item()}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'avg_val_loss': val_loss}
        return {'val_loss': val_loss, 'log': log}

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
    root_dir = "datasets/cv-corpus-5.1-2020-06-22/ru"
    train_file = "train.tsv"
    val_file = "test.tsv"
    train_set = SpeechDataset(root_dir, train_file)
    val_set = SpeechDataset(root_dir, val_file)
    net = ConvTasNet()
    speech_separation = SpeechSeparationModel(net, train_set, val_set)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(speech_separation)
