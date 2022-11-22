import torch
import torchaudio
import torchaudio.transforms as T

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import random

class SpeechConversionAudioPipeline(torch.nn.Module):
    def __init__(
        self,
        sr=16000,
        n_fft=1024,
        n_mel=128,
        win_length=1024,
        hop_length=256
    ):
        super().__init__()
        self.source_sampling_rate = sr

        pad = int((n_fft-hop_length)/2)
        self.spec = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            pad=pad, power=None,center=False, pad_mode='reflect', normalized=False, onesided=True)
        
        self.invspec = T.InverseSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length)

        # self.strech = T.TimeStretch(hop_length=hop_length, n_freq=freq)
        self.spec_aug = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=80),
            # T.TimeMasking(time_mask_param=80),
        )
        '''
        self.pitch_delta = 3
        pitch_shift_list = []
        for i in range(0, self.pitch_delta):
            delta = 2 * i + 1
            for n_step in [-delta, delta]:
                pitch_shift_list.append(
                    T.PitchShift(self.source_sampling_rate, n_step)
                )
        self.pitch_shift_modules = nn.ModuleList(pitch_shift_list)
        for m in self.pitch_shift_modules:
            m.initialize_parameters(torch.tensor([0], dtype=torch.float, device=self.device))
        '''

    def forward(self, waveform: torch.Tensor, aug: bool=False) -> torch.Tensor:
        # transform
        '''
        if random.random() <= 0.3:
            spec = waveform
        else:
            pitch_shift_index = random.choice(list(range(len(self.pitch_shift_modules))))
            spec = self.pitch_shift_modules[pitch_shift_index](waveform)
        '''

        shift_waveform = waveform
        # Convert to power spectrogram
        spec = self.spec(shift_waveform)
        # spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
        # Apply SpecAugment
        if aug:
            spec = self.spec_aug(spec)
        # Convert to wav
        wav = self.invspec(spec)

        out = torch.zeros_like(waveform)
        out[:, :, :wav.shape[2]] = wav
        return out