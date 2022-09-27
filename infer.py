import torch
import soundfile as sf
import glob

files = glob.glob("logs/lightning_logs/version_4/checkpoints/*.ckpt")
PATH = sorted(list(files))[-1]

from vits.model.vits import VITS
from vits.mel_processing import spec_to_mel_torch, mel_spectrogram_torch, spectrogram_torch
from vits.utils import load_wav_to_torch, plot_spectrogram_to_numpy
from vits import commons 
from vits.mel_processing import spectrogram_torch
from vits.utils import load_wav_to_torch, load_filepaths_and_text
from vits.text import text_to_sequence, cleaned_text_to_sequence
from vits.text.cleaners import japanese_cleaners
from vits.text.japanese import ROMAJI_LIST


model = VITS.load_from_checkpoint(PATH)
hparams = model.hparams

model.eval()

def get_text(text, hparams):
    if getattr(hparams.data, "cleaned_text", False):
        text_norm = cleaned_text_to_sequence(text)
    else:
        text_norm = text_to_sequence(text, hparams.data.text_cleaners)
    if hparams.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def split_romaji(text: str) -> str:
    out = []
    left_text = text
    while len(left_text) > 0:
        not_found = True
        for c in ROMAJI_LIST:
            if left_text.startswith(c):
                out.append(c)
                left_text = left_text[len(c):]
                not_found = False
                break
        if not_found:
            out.append(left_text[-1])
            left_text = left_text[1:]
    return " ".join(out)

source_text =  "おはようございます、皆さん." # "ココアさんの妄想です。"
import pykakasi
kks = pykakasi.kakasi()
target_text = " ".join([item['hepburn'] for item in kks.convert(source_text)])
target_text = split_romaji(target_text)
target_text = japanese_cleaners(text=target_text)
print(target_text)

text_norm = get_text(target_text, model.hparams)

text = text_norm.unsqueeze(0)
text_lengths = torch.LongTensor(1)
text_lengths[0] = text_norm.size(0)

x, x_lengths = text, text_lengths
y_hat, attn, mask, *_ = model.net_g.infer(x, x_lengths, max_len=10000)
y_hat_lengths = mask.sum([1,2]).long() * model.hparams.data.hop_length

sf.write('out.wav', y_hat[0,:,:y_hat_lengths[0]].squeeze(0).detach().numpy(), 48000, subtype='PCM_24')