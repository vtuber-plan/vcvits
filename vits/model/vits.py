
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
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(*[k for k in kwargs])

        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hparams.data.filter_length // 2 + 1,
            self.hparams.train.segment_size // self.hparams.data.hop_length,
            **self.hparams.model)
        self.net_d = MultiPeriodDiscriminator(self.hparams.model.use_spectral_norm)

        self.generator_out = None

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, x_lengths, spec, spec_lengths, y, y_lengths = batch
        
        # Generator
        if optimizer_idx == 0:
            self.generator_out = self.net_g(x, x_lengths, spec, spec_lengths)
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = self.generator_out
            y = commons.slice_segments(y, ids_slice * self.hparams.data.hop_length, self.hparams.train.segment_size) # slice

            mel = spec_to_mel_torch(
                spec, 
                self.hparams.data.filter_length, 
                self.hparams.data.n_mel_channels, 
                self.hparams.data.sampling_rate,
                self.hparams.data.mel_fmin, 
                self.hparams.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, self.hparams.train.segment_size // self.hparams.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), 
                self.hparams.data.filter_length, 
                self.hparams.data.n_mel_channels, 
                self.hparams.data.sampling_rate, 
                self.hparams.data.hop_length, 
                self.hparams.data.win_length, 
                self.hparams.data.mel_fmin, 
                self.hparams.data.mel_fmax
            )

            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.train.c_kl

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
            y = commons.slice_segments(y, ids_slice * self.hparams.data.hop_length, self.hparams.train.segment_size) # slice 
            
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

    def validation_step(self, batch, batch_idx):
        self.net_g.eval()
        
        x, x_lengths, spec, spec_lengths, y, y_lengths = batch

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]

        y_hat, attn, mask, *_ = self.net_g.infer(x, x_lengths, max_len=1000)
        y_hat_lengths = mask.sum([1,2]).long() * self.hparams.data.hop_length

        mel = spec_to_mel_torch(
            spec, 
            self.hparams.data.filter_length, 
            self.hparams.data.n_mel_channels, 
            self.hparams.data.sampling_rate,
            self.hparams.data.mel_fmin, 
            self.hparams.data.mel_fmax)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            self.hparams.data.filter_length,
            self.hparams.data.n_mel_channels,
            self.hparams.data.sampling_rate,
            self.hparams.data.hop_length,
            self.hparams.data.win_length,
            self.hparams.data.mel_fmin,
            self.hparams.data.mel_fmax
        )
        image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
        }
        audio_dict = {
        "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
        }
        # if self.global_step == 0:
        image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

        tensorboard = self.logger.experiment
        utils.summarize(
            writer=tensorboard,
            global_step=self.global_step, 
            images=image_dict,
            audios=audio_dict,
            audio_sampling_rate=self.hparams.data.sampling_rate
            )


    def configure_optimizers(self):
        
        self.optim_g = torch.optim.AdamW(
            self.net_g.parameters(), 
            self.hparams.train.learning_rate, 
            betas=self.hparams.train.betas, 
            eps=self.hparams.train.eps)
        self.optim_d = torch.optim.AdamW(
            self.net_d.parameters(),
            self.hparams.train.learning_rate, 
            betas=self.hparams.train.betas, 
            eps=self.hparams.train.eps)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g,
                            gamma=self.hparams.train.lr_decay)
        self.scheduler_g.last_epoch = self.current_epoch - 1
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, 
                            gamma=self.hparams.train.lr_decay)
        self.scheduler_d.last_epoch = self.current_epoch - 1

        return [self.optim_g, self.optim_d], [self.scheduler_g, self.scheduler_d]