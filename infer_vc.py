import json
import torch
import soundfile as sf
import glob
import os
from vits.data.dataset import coarse_f0, estimate_pitch
from vits.hparams import HParams

files = glob.glob("logs/lightning_logs/version_12/checkpoints/*.ckpt")
PATH = sorted(list(files))[-1]

from vits.model.vcvits import VCVITS
from vits.mel_processing import spec_to_mel_torch, mel_spectrogram_torch, spectrogram_torch
from vits.utils import load_wav_to_torch, plot_spectrogram_to_numpy
from vits import commons 
from vits.mel_processing import spectrogram_torch
from vits.utils import load_wav_to_torch, load_filepaths_and_text
from vits.text import text_to_sequence, cleaned_text_to_sequence
from vits.text.cleaners import japanese_cleaners
from vits.text.cleaners import chinese_cleaners
from vits.text.cleaners.japanese_mapping import ROMAJI_LIST
import torchaudio

model = VCVITS.load_from_checkpoint(PATH)
hparams = model.hparams

model.eval()

resamplers = {}

def get_audio(hparams, filename: str, sr = None):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sr is not None and sampling_rate != sr:
        # not match, then resample
        if sampling_rate in resamplers:
            resampler = resamplers[sampling_rate]
        else:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=sr)
            resamplers[sampling_rate] = resampler
        audio = resampler(audio)
        sampling_rate = sr
        # raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)

    # load spec
    spec = spectrogram_torch(audio_norm, hparams.filter_length, sr, hparams.hop_length, hparams.win_length, center=False)
    spec = torch.squeeze(spec, 0)


    # load mel
    melspec = spec_to_mel_torch(spec, hparams.filter_length, hparams.n_mel_channels, sr, hparams.mel_fmin, hparams.mel_fmax)
    melspec = torch.squeeze(melspec, 0)

    # load pitch
    pitch_mel = estimate_pitch(
        audio=audio.numpy(), sr=sampling_rate, n_fft=hparams.filter_length,
        win_length=hparams.win_length, hop_length=320, method='pyin',
        normalize_mean=None, normalize_std=None, n_formants=1)
    
    coarse_pitch = coarse_f0(pitch_mel)
    pitch_mel = coarse_pitch

    return spec, audio_norm, melspec, pitch_mel

def convert(source_audio, target_audio):
    with open("configs/paimoon_base_vc.json", "r") as f:
        data = f.read()
    config = json.loads(data)
    
    hparams = HParams(**config)

    x_spec, x_wav, x_melspec, x_pitch = get_audio(hparams.data, source_audio, sr=16000)

    x_wav = x_wav.unsqueeze(0)
    x_wav_lengths = torch.tensor([x_wav.shape[2]], dtype=torch.long)
    x_pitch = x_pitch.long()
    x_pitch_lengths = torch.tensor([x_pitch.shape[1]], dtype=torch.long)
    
    y_hat, attn, mask, *_ = model.net_g.infer(x_wav, x_wav_lengths, x_pitch, x_pitch_lengths, max_len=10000, length_scale=1)
    y_hat_lengths = mask.sum([1,2]).long() * model.hparams.data.hop_length

    sf.write(target_audio, y_hat[0,:,:y_hat_lengths[0]].squeeze(0).detach().numpy(), 48000, subtype='PCM_24')

convert("dataset/ChinoCorpus/CN0B4000.wav", 'out.wav')