import math
import time
import os
import random
import numpy as np
import librosa
from librosa import pyin
# from natsupitch.core import pyin
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torchaudio

from .. import commons 
from vits.mel_processing import MAX_WAV_VALUE, mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from vits.utils import load_filepaths, load_wav_to_torch, load_filepaths_and_text
from ..text import text_to_sequence, cleaned_text_to_sequence
from ..utils import load_wav_to_torch

def normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None]
    pitch /= std[:, None]
    pitch[zeros] = 0.0
    return pitch

def estimate_pitch(audio: np.ndarray, sr: int, mel_len: int, n_fft: int, win_length: int, hop_length: int,
                    method='pyin', normalize_mean=None, normalize_std=None, n_formants=1):
    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)

    if method == 'pyin':
        snd, sr = audio, sr
        pad_size = int((n_fft-hop_length)/2)
        snd = np.pad(snd, (pad_size, pad_size), mode='reflect')

        pitch_mel, voiced_flag, voiced_probs = pyin(
            snd,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=win_length,
            hop_length=hop_length,
            center=False,
            pad_mode='reflect')
        assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

        pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
        pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

        if n_formants > 1:
            raise NotImplementedError
    else:
        raise ValueError

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel

class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text: str, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.hparams = hparams
        self.text_cleaners  = hparams.text_cleaners
        self.max_wav_value  = hparams.max_wav_value
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        to_tensor = lambda x: torch.Tensor([x]) if type(x) is float else x
        self.pitch_mean=to_tensor(214.72203)
        self.pitch_std=to_tensor(65.72038)

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav, melspec, pitch_mel = self.get_audio(audiopath)
        energy = torch.norm(melspec.float(), dim=0, p=2).unsqueeze(0)
        return {
            "text": text,
            "spec": spec,
            "wav": wav,
            "melspec": melspec,
            "pitch": pitch_mel,
            "energy": energy,
        }

    def get_audio(self, filename: str):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)

        # load spec
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length, self.sampling_rate,
                    self.hop_length, self.win_length, center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        
        # load mel
        mel_filename = filename.replace(".wav", ".mel.pt")
        if os.path.exists(mel_filename):
            melspec = torch.load(mel_filename)
        else:
            melspec = spec_to_mel_torch(spec, self.hparams.filter_length, self.hparams.n_mel_channels, 
                                self.hparams.sampling_rate, self.hparams.mel_fmin, self.hparams.mel_fmax)
            melspec = torch.squeeze(melspec, 0)
            torch.save(melspec, mel_filename)

        # load pitch
        pitch_filename = filename.replace(".wav", ".pitch.pt")
        if os.path.exists(pitch_filename):
            pitch_mel = torch.load(pitch_filename)
        else:
            pitch_mel = estimate_pitch(
                audio=audio.numpy(), sr=sampling_rate, mel_len=melspec.shape[-1], n_fft=self.hparams.filter_length,
                win_length=self.hparams.win_length, hop_length=self.hparams.hop_length, method='pyin',
                normalize_mean=None, normalize_std=None, n_formants=1)
            pitch_mel = np.log2(pitch_mel + 1e-6)
            pitch_mel = torch.FloatTensor(pitch_mel)
            torch.save(pitch_mel, pitch_filename)

        return spec, audio_norm, melspec, pitch_mel

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        return {
            "text": text,
            "spec": spec,
            "wav": wav,
            "sid": sid,
        }

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)

class AnyVoiceConversionLoader(torch.utils.data.Dataset):
    def __init__(self, audiopaths: str, hparams):
        self.audiopaths = load_filepaths(audiopaths)
        self.hparams = hparams
        self.text_cleaners  = hparams.text_cleaners
        self.max_wav_value  = hparams.max_wav_value
        self.source_sampling_rate  = hparams.source_sampling_rate
        self.target_sampling_rate  = hparams.target_sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        self.resamplers = {}

        random.seed(1234)
        random.shuffle(self.audiopaths)

    def get_audio(self, filename: str, sr = None):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sr is not None and sampling_rate != sr:
            # not match, then resample
            if sampling_rate in self.resamplers:
                resampler = self.resamplers[sampling_rate]
            else:
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=sr)
                self.resamplers[sampling_rate] = resampler
            audio = resampler(audio)
            sampling_rate = sr
            # raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)

        # load spec
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length, sr, self.hop_length, self.win_length, center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        
        # load mel
        mel_filename = filename.replace(".wav", ".mel.pt")
        if os.path.exists(mel_filename):
            melspec = torch.load(mel_filename)
        else:
            melspec = spec_to_mel_torch(spec, self.hparams.filter_length, self.hparams.n_mel_channels, sr, self.hparams.mel_fmin, self.hparams.mel_fmax)
            melspec = torch.squeeze(melspec, 0)
            torch.save(melspec, mel_filename)

        # load pitch
        pitch_filename = filename.replace(".wav", ".pitch.pt")
        if os.path.exists(pitch_filename):
            pitch_mel = torch.load(pitch_filename)
        else:
            pitch_mel = estimate_pitch(
                audio=audio.numpy(), sr=sampling_rate, mel_len=melspec.shape[-1], n_fft=self.hparams.filter_length,
                win_length=self.hparams.win_length, hop_length=self.hparams.hop_length, method='pyin',
                normalize_mean=None, normalize_std=None, n_formants=1)
            pitch_mel = np.log10(pitch_mel + 1e-6)
            pitch_mel = torch.FloatTensor(pitch_mel)
            torch.save(pitch_mel, pitch_filename)

        return spec, audio_norm, melspec, pitch_mel

    def __getitem__(self, index):
        audiopath = self.audiopaths[index]
        x_spec, x_wav, x_melspec, x_pitch_mel = self.get_audio(audiopath, self.source_sampling_rate)
        y_spec, y_wav, y_melspec, y_pitch_mel = self.get_audio(audiopath, self.target_sampling_rate)
        return {
            "x_spec": x_spec,
            "x_wav": x_wav,
            "x_mel": x_melspec,
            "x_pitch": x_pitch_mel,

            "y_spec": y_spec,
            "y_wav": y_wav,
            "y_mel": y_melspec,
            "y_pitch": y_pitch_mel,
        }

    def __len__(self):
        return len(self.audiopaths)
