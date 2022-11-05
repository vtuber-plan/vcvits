
import math
import time
import os
import random
from typing import Optional
import numpy as np
import librosa
from librosa import pyin
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torchaudio
 
from vits.mel_processing import MAX_WAV_VALUE, mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from vits.utils import load_filepaths, load_wav_to_torch, load_filepaths_and_text
from ..audio import coarse_f0, estimate_pitch, get_audio, get_pitch, load_audio, shift_audio
import tqdm

import random
from joblib import Memory
import warnings

class VoiceConversionMultiSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, audiopaths: str, hparams, cache_dir: Optional[str]):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.hparams = hparams
        self.max_wav_value  = hparams.max_wav_value
        self.source_sampling_rate  = hparams.source_sampling_rate
        self.target_sampling_rate  = hparams.target_sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length

        self.resamplers = {}

        random.seed(1234)
        random.shuffle(self.audiopaths)

        # Define effects
        self.effects = [
            ["lowpass", "-1", "300"], # apply single-pole lowpass filter
        ]

        self.cache_dir = cache_dir
        self.memory = Memory(self.cache_dir, verbose=0)
        if cache_dir is not None:
            self.load_audio = self.memory.cache(load_audio)
            self.shift_audio = shift_audio
            self.get_audio = get_audio
            self.get_pitch = self.memory.cache(get_pitch)

    def get_item(self, index: int, pitch_shift: int = 0, apply_effect: bool=False):
        item = self.audiopaths[index]
        audiopath = item[0]
        if len(item) == 1:
            sid = 0
        else:
            sid = int(item[1])
        
        x_audio = self.load_audio(audiopath, sr=self.source_sampling_rate)
        x_shifted_audio = self.shift_audio(x_audio, sr=self.source_sampling_rate, pitch_shift = pitch_shift)

        # Apply effects
        if apply_effect:
            x_shifted_audio, new_sample_rate = torchaudio.sox_effects.apply_effects_tensor(x_shifted_audio, self.source_sampling_rate, self.effects)

        x_wav = x_shifted_audio.unsqueeze(0)

        x_spec, x_melspec = self.get_audio(
            x_wav,
            max_wav_value = self.max_wav_value,
            filter_length = self.filter_length,
            hop_length = self.hop_length,
            win_length = self.win_length,
            n_mel_channels = self.hparams.n_mel_channels,
            mel_fmin = self.hparams.mel_fmin,
            mel_fmax = self.hparams.mel_fmax,
            hubert_channels = self.hparams.hubert_channels,
            num_pitch = self.hparams.num_pitch,
            sr=self.source_sampling_rate,
            load_spec=False,
            load_mel=False)
        x_pitch_mel = self.get_pitch(
            audiopath,
            self.hparams.filter_length,
            self.hparams.win_length,
            self.hparams.num_pitch,
            self.source_sampling_rate
        )

        y_audio = self.load_audio(audiopath, sr=self.target_sampling_rate)
        y_wav = y_audio.unsqueeze(0)

        y_spec, y_melspec = self.get_audio(
            y_wav,
            max_wav_value = self.max_wav_value,
            filter_length = self.filter_length,
            hop_length = self.hop_length,
            win_length = self.win_length,
            n_mel_channels = self.hparams.n_mel_channels,
            mel_fmin = self.hparams.mel_fmin,
            mel_fmax = self.hparams.mel_fmax,
            hubert_channels = self.hparams.hubert_channels,
            num_pitch = self.hparams.num_pitch,
            sr=self.target_sampling_rate,
            load_spec=True,
            load_mel=False)
        return {
            "sid": sid,

            "x_wav": x_wav,
            "x_pitch": x_pitch_mel,

            "y_spec": y_spec,
            "y_wav": y_wav,
        }

    def __getitem__(self, index):
        if random.random() < 0.3:
            pitch_shift = 0
        else:
            pitch_shift = random.randint(-12, 12)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ret = self.get_item(index, pitch_shift)
        return ret

    def __len__(self):
        return len(self.audiopaths)
