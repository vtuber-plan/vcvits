import os
import torch
import torchaudio
import fairseq
import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from vits.data.audio import coarse_f0, estimate_pitch
from vits.mel_processing import spec_to_mel_torch, spectrogram_torch

from joblib import Parallel, delayed

import vits.utils as utils

hubert = None

def load_hubert(path: str, device: str):
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
    hubert = models[0].to(device)
    hubert.eval()
    return hubert

def preprocess_single(hparams, filename: str, sr=16000, load_features: bool = False, device: str="cpu"):
    audio, sampling_rate = utils.load_wav_to_torch(filename)
    if sr is not None and sampling_rate != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=sr)
        audio = resampler(audio)
        sampling_rate = sr
        # raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.sampling_rate))
    audio_norm = audio / hparams.data.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)

    # load spec
    spec_filename = filename.replace(".wav", f"_{sampling_rate}.spec.pt")
    if not os.path.exists(spec_filename):
        spec = spectrogram_torch(audio_norm, hparams.data.filter_length, sr, hparams.data.hop_length, hparams.data.win_length, center=False)
        spec = torch.squeeze(spec, 0)
        torch.save(spec, spec_filename)
    
    # load mel
    mel_filename = filename.replace(".wav", f"_{sampling_rate}.mel.pt")
    if not os.path.exists(mel_filename):
        melspec = spec_to_mel_torch(spec, hparams.data.filter_length, hparams.data.n_mel_channels, sr, hparams.data.mel_fmin, hparams.data.mel_fmax)
        melspec = torch.squeeze(melspec, 0)
        torch.save(melspec, mel_filename)

    # load pitch
    pitch_filename = filename.replace(".wav", f"_{sampling_rate}.pitch.pt")
    if not os.path.exists(pitch_filename):
        pitch_mel = estimate_pitch(
            audio=audio.numpy(), sr=sampling_rate, n_fft=hparams.data.filter_length,
            win_length=hparams.data.win_length, hop_length=320, method='pyin',
            normalize_mean=None, normalize_std=None, n_formants=1)
        
        coarse_pitch = coarse_f0(pitch_mel)
        pitch_mel = coarse_pitch
        torch.save(pitch_mel, pitch_filename)

    # features
    feature_filename = filename.replace(".wav", f"_{sampling_rate}.feature.pt")
    if not os.path.exists(feature_filename) and load_features:
        global hubert
        wav = F.pad(audio_norm, ((400 - 320) // 2, (400 - 320) // 2))
        wav_input = wav.squeeze(1).to(device)

        if hubert is None:
            hubert = load_hubert(hparams.data.hubert_ckpt, device)

        hubert_features, _ = hubert.extract_features(wav_input)
        hubert_features = hubert_features.transpose(1, -1).squeeze(0)

        hubert_features = hubert_features.to("cpu")
        torch.save(hubert_features, feature_filename)

def preprocess(hparams, files, sr=16000, load_features: bool = False):
    # load hubert
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    audiopaths = utils.load_filepaths_and_text(files)
    audiopaths = [item[0] for item in audiopaths]

    audiopaths = sorted(audiopaths, reverse=True)

    print("Preprocessing dataset...")
    with torch.inference_mode():
        Parallel(n_jobs=32, backend="loky")\
            (delayed(preprocess_single)(hparams, filename, sr, False, device) \
                for filename in tqdm.tqdm(audiopaths))
    
    if load_features:
        for filename in tqdm.tqdm(audiopaths):
            preprocess_single(hparams, filename, sr, load_features, device)