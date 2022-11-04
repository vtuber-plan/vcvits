
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
from ..audio import get_audio_preload
import tqdm

from joblib import Memory
import warnings
import random

class PreloadAnyVoiceConversionMultiSpeakerDataset(torch.utils.data.Dataset):
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

        self.cache_dir = cache_dir
        if cache_dir is not None:
            self.memory = Memory(self.cache_dir, verbose=0)
            self.get_item_cached = self.memory.cache(self.get_item)

    def get_item(self, index: int, pitch_shift: int = 0):
        item = self.audiopaths[index]
        audiopath = item[0]
        if len(item) == 1:
            sid = 0
        else:
            sid = int(item[1])
        x_spec, x_wav, x_melspec, x_pitch_mel, x_hubert_features = get_audio_preload(
            audiopath,
            max_wav_value = self.max_wav_value,
            filter_length = self.filter_length,
            hop_length = self.hop_length,
            win_length = self.win_length,
            n_mel_channels = self.hparams.n_mel_channels,
            mel_fmin = self.hparams.mel_fmin,
            mel_fmax = self.hparams.mel_fmax,
            hubert_channels = self.hparams.hubert_channels,
            num_pitch = self.hparams.num_pitch,
            pitch_shift = pitch_shift,
            sr=self.source_sampling_rate, load_features=True)
        y_spec, y_wav, y_melspec, y_pitch_mel, y_hubert_features = get_audio_preload(
            audiopath,
            max_wav_value = self.max_wav_value,
            filter_length = self.filter_length,
            hop_length = self.hop_length,
            win_length = self.win_length,
            n_mel_channels = self.hparams.n_mel_channels,
            mel_fmin = self.hparams.mel_fmin,
            mel_fmax = self.hparams.mel_fmax,
            hubert_channels = self.hparams.hubert_channels,
            num_pitch = self.hparams.num_pitch,
            sr=self.target_sampling_rate, load_features=False)
        return {
            "sid": sid,

            "x_spec": x_spec,
            "x_wav": x_wav,
            "x_mel": x_melspec,
            "x_pitch": x_pitch_mel,
            "x_hubert_features": x_hubert_features,

            "y_spec": y_spec,
            "y_wav": y_wav,
            "y_mel": y_melspec,
            "y_pitch": y_pitch_mel,
            "y_hubert_features": y_hubert_features,
        }

    def __getitem__(self, index):
        if random.random() < 0.3:
            pitch_shift = 0
        else:
            pitch_shift = random.randint(-12, 12)

        if hasattr(self, "get_item_cached"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.get_item_cached(index, pitch_shift)
        else:
            return self.get_item(index, pitch_shift)

    def __len__(self):
        return len(self.audiopaths)

class MemoryPreloadAnyVoiceConversionMultiSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, audiopaths: str, hparams):
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
        self.dataset = []
        self.preload()
    
    def preload(self):
        for index, item in enumerate(tqdm.tqdm(self.audiopaths)):
            audiopath = item[0]
            if len(item) == 1:
                sid = 0
            else:
                sid = int(item[1])
            x_spec, x_wav, x_melspec, x_pitch_mel, x_hubert_features = get_audio_preload(
                audiopath,
                max_wav_value = self.max_wav_value,
                filter_length = self.filter_length,
                hop_length = self.hop_length,
                win_length = self.win_length,
                n_mel_channels = self.hparams.n_mel_channels,
                mel_fmin = self.hparams.mel_fmin,
                mel_fmax = self.hparams.mel_fmax,
                hubert_channels = self.hparams.hubert_channels,
                num_pitch = self.hparams.num_pitch,
                sr=self.source_sampling_rate, load_features=True)
            y_spec, y_wav, y_melspec, y_pitch_mel, y_hubert_features = get_audio_preload(
                audiopath,
                max_wav_value = self.max_wav_value,
                filter_length = self.filter_length,
                hop_length = self.hop_length,
                win_length = self.win_length,
                n_mel_channels = self.hparams.n_mel_channels,
                mel_fmin = self.hparams.mel_fmin,
                mel_fmax = self.hparams.mel_fmax,
                hubert_channels = self.hparams.hubert_channels,
                num_pitch = self.hparams.num_pitch,
                sr=self.target_sampling_rate, load_features=False)
            self.dataset.append(
                {
                    "sid": sid,

                    "x_spec": x_spec,
                    "x_wav": x_wav,
                    "x_mel": x_melspec,
                    "x_pitch": x_pitch_mel,
                    "x_hubert_features": x_hubert_features,

                    "y_spec": y_spec,
                    "y_wav": y_wav,
                    "y_mel": y_melspec,
                    "y_pitch": y_pitch_mel,
                    "y_hubert_features": y_hubert_features,
                }
            )

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.audiopaths)

