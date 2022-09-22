
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


import pytorch_lightning as pl

from .synthesizer_trn import SynthesizerTrn
from .multi_period_discriminator import MultiPeriodDiscriminator
from ..text.symbols import symbols
from ..mel_processing import spec_to_mel_torch, mel_spectrogram_torch
from .losses import discriminator_loss
from .. import commons

class VITS(pl.LightningModule):
    def __init__(self, hps):
        super().__init__()
        
        self.net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        self.net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
        self.hps = hps

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, x_lengths, spec, spec_lengths, y, y_lengths = batch

        y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(x, x_lengths, spec, spec_lengths)

        mel = spec_to_mel_torch(
            spec, 
            self.hps.data.filter_length, 
            self.hps.data.n_mel_channels, 
            self.hps.data.sampling_rate,
            self.hps.data.mel_fmin, 
            self.hps.data.mel_fmax)
        y_mel = commons.slice_segments(mel, ids_slice, self.hps.train.segment_size // self.hps.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1), 
            self.hps.data.filter_length, 
            self.hps.data.n_mel_channels, 
            self.hps.data.sampling_rate, 
            self.hps.data.hop_length, 
            self.hps.data.win_length, 
            self.hps.data.mel_fmin, 
            self.hps.data.mel_fmax
        )

        y = commons.slice_segments(y, ids_slice * self.hps.data.hop_length, self.hps.train.segment_size) # slice 

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc


        # Logging to TensorBoard by default
        self.log("losses_disc_r", losses_disc_r)
        self.log("losses_disc_g", losses_disc_g)
        self.log("loss_disc_all", loss_disc_all)
        return loss_disc_all

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
