
import pytorch_lightning as pl

from .synthesizer_trn import SynthesizerTrn
from .multi_period_discriminator import MultiPeriodDiscriminator
from ..text.symbols import symbols

class VITS(pl.LightningModule):
    def __init__(self, hps):
        super().__init__()
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
