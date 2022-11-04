
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
from ..vocoder import Generator


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
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
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

        self.use_sdp = use_sdp

        self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

        self.enc_p = HubertContentEncoder(kwargs["hubert_ckpt"], inter_channels, hidden_channels, filter_channels,
                                n_heads, n_layers, kernel_size, p_dropout)
        self.dec = Generator(inter_channels,
                             resblock,
                             resblock_kernel_sizes,
                             resblock_dilation_sizes,
                             upsample_rates,
                             upsample_initial_channel,
                             upsample_kernel_sizes,
                             gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        # self.duration_predictor = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)
        self.duration_predictor = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)

        if n_speakers >= 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x_wav, x_wav_lengths, x_pitch, x_pitch_lengths, y_spec, y_spec_lengths, sid=None):
        # x: [batch, text_max_length]
        # text encoding
        x, m_p, logs_p, x_mask = self.enc_p(x_wav, x_wav_lengths, x_pitch, x_pitch_lengths)

        # m_p, logs_p, 
        if self.n_speakers >= 1:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(y_spec, y_spec_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            # [b, 1, t_s]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
            # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        # attn： [batch, 1, audio_max_frame, text_max_length] 
        w = attn.sum(2)

        # logw_ = torch.log(w + 1e-6) * x_mask
        # logw = self.duration_predictor(x, x_mask, g=g)
        # l_length = torch.sum((logw - logw_)**2, [1, 2]) / torch.sum(x_mask)  # for averaging
        l_length = self.duration_predictor(x, x_mask, w, g=g)
        l_length = l_length / torch.sum(x_mask)
    
        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_spec_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return o, l_length, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, x_pitch, x_pitch_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, x_pitch, x_pitch_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        y_lengths = (x_lengths * length_scale).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)

        y_max_len = torch.max(y_lengths).item()
        m_p = F.interpolate(m_p, size=(y_max_len,), mode="linear")
        logs_p = F.interpolate(logs_p, size=(y_max_len,), mode="linear")

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
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
