
import math
import torchaudio
import torch
from torch import nn
from torch.nn import functional as F

import vits.commons as commons

from ..encoders.content_encoder import HubertContentEncoder
from ..encoders.posterior_encoder import PosteriorEncoder
from ..flow import ResidualCouplingBlock
from ..predictors.duration_predictor import StochasticDurationPredictor, DurationPredictor
from ..predictors.pitch_predictor import PitchPredictor
from ..predictors.energy_predictor import EnergyPredictor


class SynthesizerSVC(nn.Module):
    def __init__(self, spec_channels, segment_size,
                 inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 hubert_channels,
                 num_pitch,
                 n_speakers=0,
                 gin_channels=0,
                 **kwargs):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.hubert_channels = hubert_channels
        self.num_pitch = num_pitch

        self.enc_p = HubertContentEncoder(kwargs["hubert_ckpt"], inter_channels, hidden_channels, filter_channels,
                                n_heads, n_layers, kernel_size, p_dropout, hubert_channels, num_pitch)
        self.dec = torch.hub.load("vtuber-plan/hifi-gan:v0.3.0", "hifigan_48k")
        for param in self.dec.parameters():
            param.requires_grad = False
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        # self.duration_predictor = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)
        # self.duration_predictor = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        if n_speakers >= 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x_wav, x_wav_lengths, x_pitch, x_pitch_lengths, y_spec, y_spec_lengths, sid=None):
        # x: [batch, text_max_length]
        x, m_p, logs_p, x_mask = self.enc_p(x_wav, x_wav_lengths, x_pitch, x_pitch_lengths)

        # m_p, logs_p, 
        if self.n_speakers >= 1:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(y_spec, y_spec_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        m_p = F.interpolate(m_p, size=(y_spec.shape[2],), mode="nearest")
        logs_p = F.interpolate(logs_p, size=(y_spec.shape[2],), mode="nearest")

        z_slice, ids_slice = commons.rand_slice_segments(z, y_spec_lengths, self.segment_size)
        o = self.dec(z_slice)
        return o, ids_slice, z_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, x_pitch, x_pitch_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, x_pitch, x_pitch_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        y_lengths = (x_lengths * length_scale).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)

        y_max_len = torch.max(y_lengths).item()
        m_p = F.interpolate(m_p, size=(y_max_len,), mode="nearest")
        logs_p = F.interpolate(logs_p, size=(y_max_len,), mode="nearest")

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # Up dim to march vocoder
        o = self.dec((z * y_mask)[:, :, :max_len])
        return o, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
