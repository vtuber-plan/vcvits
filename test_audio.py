import torch
import torchaudio

import torchaudio.transforms as T

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import random

n_fft=2048
hop_length=512
win_length=2048

pad = int((n_fft-hop_length)/2)
spec = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
    pad=pad, power=None,center=False, pad_mode='reflect', normalized=False, onesided=True)

invspec = T.InverseSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
)

audio, sr = torchaudio.load("test.wav")

s = spec(audio)
wav = invspec(s)

torchaudio.save("out.wav", wav, sr)