import torch
import torchaudio

audio, sr = torchaudio.load("dataset/example/chino/CN0A3008.wav")

effects = [
    ["lowpass", "-1", "300"], # apply single-pole lowpass filter
]

shifted_audio = torchaudio.functional.pitch_shift(audio, sr, -8)
x_shifted_audio, new_sample_rate = torchaudio.sox_effects.apply_effects_tensor(shifted_audio, sr, effects)

print(audio.dtype)
print(audio.shape)

torchaudio.save("out.wav", x_shifted_audio, sr)