
import math
import time
import os
import random
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
from ..audio import get_audio

class PreloadAnyVoiceConversionMultiSpeakerLoader(torch.utils.data.Dataset):
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

    def __getitem__(self, index):
        item = self.audiopaths[index]
        audiopath = item[0]
        if len(item) == 1:
            sid = 0
        else:
            sid = int(item[1])
        x_spec, x_wav, x_melspec, x_pitch_mel, x_hubert_features = get_audio(
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
        y_spec, y_wav, y_melspec, y_pitch_mel, y_hubert_features = get_audio(
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

    def __len__(self):
        return len(self.audiopaths)
