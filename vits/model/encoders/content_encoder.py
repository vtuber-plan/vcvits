
import math
import torch
from torch import nn
from torch.nn import functional as F
import torch

from vits.model.transformer.relative_attention_transformer import TransformerEncoder

import fairseq
from ... import commons

class HubertContentEncoder(nn.Module):
    def __init__(self,
                hubert_ckpt: str,
                out_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                n_fft=2048,
                hop_size=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.out_channels = out_channels
        
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([hubert_ckpt])
        self.hubert = models[0]
        for param in self.hubert.parameters():
            param.requires_grad = False
        
        self.emb_pitch = nn.Embedding(512, hidden_channels)
        nn.init.normal_(self.emb_pitch.weight, 0.0, hidden_channels ** -0.5)
        
        self.encoder = TransformerEncoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, pitch, pitch_lengths):
        wav = F.pad(x, ((400 - 320) // 2, (400 - 320) // 2))
        x_encoded, _ = self.hubert.extract_features(wav.squeeze(1))
        hubert_out = x_encoded.transpose(1, -1)
        
        pitch_out = self.emb_pitch(pitch).transpose(1, -1)
        out = hubert_out + pitch_out

        # n_downsample = self.hubert.feature_extractor.downsample_num
        x_mask = torch.unsqueeze(commons.sequence_mask((x_lengths/320).int(), out.size(2)), 1).to(x.dtype)
   
        x_out = self.encoder(out * x_mask, x_mask)

        stats = self.proj(x_out) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x_out, m, logs, x_mask


class PreloadHubertContentEncoder(nn.Module):
    def __init__(self,
                out_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                hubert_channels,
                num_pitch,
                n_fft=2048,
                hop_size=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.out_channels = out_channels

        proj_channels = hidden_channels // 2
        
        self.hubert_proj = nn.Linear(hubert_channels, proj_channels)
        self.emb_pitch = nn.Embedding(num_pitch, proj_channels)
        nn.init.normal_(self.emb_pitch.weight, 0.0, proj_channels ** -0.5)
        self.pitch_proj = nn.Linear(proj_channels, proj_channels)
        
        self.encoder = TransformerEncoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, pitch, pitch_lengths):
        hubert_out = self.hubert_proj(x.transpose(1, -1)).transpose(1, -1)
        
        pitch_embeddings = self.emb_pitch(pitch).transpose(1, -1)
        pitch_out = self.pitch_proj(pitch_embeddings.transpose(1, -1)).transpose(1, -1)

        out = torch.concat((hubert_out, pitch_out), dim=1)

        # n_downsample = self.hubert.feature_extractor.downsample_num
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths.int(), out.size(2)), 1).to(x.dtype)
   
        x_out = self.encoder(out * x_mask, x_mask)

        stats = self.proj(x_out) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x_out, m, logs, x_mask
