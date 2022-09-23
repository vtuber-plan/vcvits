
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


import pytorch_lightning as pl

from .synthesizer_trn import SynthesizerTrn
from .multi_period_discriminator import MultiPeriodDiscriminator
from ..text.symbols import symbols
from ..mel_processing import spec_to_mel_torch, mel_spectrogram_torch
from .losses import discriminator_loss, kl_loss,feature_loss, generator_loss
from .. import commons
from .. import utils

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
        self.epoch_str = 1

        self.generator_out = None

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, x_lengths, spec, spec_lengths, y, y_lengths = batch
        
        # Generator
        if optimizer_idx == 0:
            self.generator_out = self.net_g(x, x_lengths, spec, spec_lengths)
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = self.generator_out
            y = commons.slice_segments(y, ids_slice * self.hps.data.hop_length, self.hps.train.segment_size) # slice

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

            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hps.train.c_kl

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            grad_norm_g = commons.clip_grad_value_(self.net_g.parameters(), None)

            # Logging to TensorBoard by default
            lr = self.optim_g.param_groups[0]['lr']
            scalar_dict = {"loss/g/total": loss_gen_all, "learning_rate": lr, "grad_norm_g": grad_norm_g}
            scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

            scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})

            '''
            image_dict = { 
                "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
                "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
            }
            '''
            image_dict = {}
            
            tensorboard = self.logger.experiment
            utils.summarize(
                writer=tensorboard,
                global_step=self.global_step, 
                images=image_dict,
                scalars=scalar_dict)
            return loss_gen_all
    
        # Discriminator
        if optimizer_idx == 1:
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = self.generator_out
            y = commons.slice_segments(y, ids_slice * self.hps.data.hop_length, self.hps.train.segment_size) # slice 
            
            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc

            grad_norm_d = commons.clip_grad_value_(self.net_d.parameters(), None)

            # log

            lr = self.optim_g.param_groups[0]['lr']
            scalar_dict = {"loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d}
            scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
            scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

            image_dict = {}
            
            tensorboard = self.logger.experiment

            utils.summarize(
                writer=tensorboard,
                global_step=self.global_step, 
                images=image_dict,
                scalars=scalar_dict)

            return loss_disc_all


    def configure_optimizers(self):
        
        self.optim_g = torch.optim.AdamW(
            self.net_g.parameters(), 
            self.hps.train.learning_rate, 
            betas=self.hps.train.betas, 
            eps=self.hps.train.eps)
        self.optim_d = torch.optim.AdamW(
            self.net_d.parameters(),
            self.hps.train.learning_rate, 
            betas=self.hps.train.betas, 
            eps=self.hps.train.eps)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=self.hps.train.lr_decay, last_epoch=self.epoch_str-2)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=self.hps.train.lr_decay, last_epoch=self.epoch_str-2)

        return [self.optim_g, self.optim_d], [self.scheduler_g, self.scheduler_d]