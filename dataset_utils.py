import matplotlib.pyplot as plt
import torch.utils.data
import torch.distributions
import numpy as np
from librosa.filters import mel as librosa_mel_fn
import librosa
import librosa.display



def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
        y,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
        center=False,
        use_full_spec=False,
        return_mel_and_spec=False,
):
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).unsqueeze(0)

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    result = spectral_normalize_torch(spec)

    if not use_full_spec:
        mel = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
        mel = spectral_normalize_torch(mel)
        result = mel.squeeze()

        if return_mel_and_spec:
            spec = spectral_normalize_torch(spec)
            return result, spec
    return result



def draw_spec(x, title, sr, ax, mel=False, compute=True):
    try:
       plt.sca(ax)
       n_fft=1024
       num_mels=80
       hop_size=256
       win_size=1024
       fmin=0
       fmax=8000
       if compute:
           if mel:
               f = mel_spectrogram(
                   x,
                   n_fft=n_fft,
                   num_mels=num_mels,
                   sampling_rate=sr,
                   hop_size=hop_size,
                   win_size=win_size,
                   fmin=fmin,
                   fmax=fmax,
               )
               f = f.squeeze().numpy()
           else:
               f = librosa.core.stft(x, n_fft=n_fft, hop_length=hop_size)
               f = np.abs(f)
       else:
           f = x
       librosa.display.specshow(
           librosa.amplitude_to_db(f, ref=np.max),
           y_axis="hz",
           x_axis="time",
           sr=2 * fmax if mel else sr,
           hop_length=hop_size,
       )
       plt.colorbar(format="%+2.0f dB")
       plt.title(title)
       plt.tight_layout()
    except:
       pass