__all__ = ["MOSNet", "LSD", "SiSNR", "SiSPNR", "STOI", "PESQ", "ScaleInvariantSignalToDistortionRatio"]

import os
from abc import ABC, abstractmethod

import hydra.utils
import torchaudio
import torch
import numpy as np
from hydra.core.hydra_config import HydraConfig
from pesq import pesq
from pystoi import stoi

from metric_nets import Wav2Vec2MOS


class Metric(ABC):
    name = "Abstract Metric"

    def __init__(self, num_splits=5, device="cuda", big_val_size=500):
        self.num_splits = num_splits
        self.device = device
        self.val_size = None
        self.result = dict()
        self.big_val_size = big_val_size

    @abstractmethod
    def better(self, first, second):
        pass

    @abstractmethod
    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        pass

    def compute(self, samples, real_samples, epoch_num, epoch_info):
        self._compute(samples, real_samples, epoch_num, epoch_info)
        self.result["val_size"] = samples.shape[0]

        if "best_mean" not in self.result or self.better(
            self.result["mean"], self.result["best_mean"]
        ):
            self.result["best_mean"] = self.result["mean"]
            self.result["best_std"] = self.result["std"]
            self.result["best_epoch"] = epoch_num

    def save_result(self, epoch_info):
        metric_name = self.name
        for key, value in self.result.items():
            epoch_info[f"metrics_{key}/{metric_name}"] = value


def get_power_spectrum(wav, n_fft: int, hop_length: int):
    spectrum = torch.stft(wav.view(wav.size(0), -1), n_fft=n_fft, hop_length=hop_length, return_complex=True).abs()
    return spectrum.transpose(-1, -2)  # (B, T, F)


class LSD(Metric):
    name = "LSD"

    def __init__(self, sampling_rate: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr = sampling_rate
        self.hop_length = int(self.sr / 100)
        self.n_fft = int(2048 / (44100 / self.sr))

    def better(self, first, second):
        return first < second

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        spectrum_hat = get_power_spectrum(samples, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrum = get_power_spectrum(real_samples, n_fft=self.n_fft, hop_length=self.hop_length)

        lsd = torch.log10(spectrum ** 2 / (spectrum_hat ** 2 + 1e-9) + 1e-9) ** 2
        lsd = torch.sqrt(lsd.mean(-1)).mean(-1)

        lsd = lsd.cpu().numpy()
        self.result["mean"] = np.mean(lsd)
        self.result["std"] = np.std(lsd)


class ScaleInvariantSignalToDistortionRatio(Metric):
    """
    See https://arxiv.org/pdf/1811.02508.pdf
    """

    name = "SISDR"

    def better(self, first, second):
        return first > second

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        real_samples, samples = real_samples.squeeze(), samples.squeeze()
        if real_samples.dim() == 1:
            real_samples = real_samples[None]
            samples = samples[None]
        alpha = (samples * real_samples).sum(
            dim=1, keepdim=True
        ) / real_samples.square().sum(dim=1, keepdim=True)
        real_samples_scaled = alpha * real_samples
        e_target = real_samples_scaled.square().sum(dim=1)
        e_res = (samples - real_samples_scaled).square().sum(dim=1)
        si_sdr = 10 * torch.log10(e_target / e_res).cpu().numpy()

        self.result["mean"] = np.mean(si_sdr)
        self.result["std"] = np.std(si_sdr)


class SiSNR(Metric):
    name = "SiSNR"

    def better(self, first, second):
        return first > second

    def _compute(self, samples, real_samples, epoch_num, epoch_info):

        alpha = (samples * real_samples).sum(-1, keepdims=True) / (real_samples.square().sum(-1, keepdims=True) + 1e-9)
        real_samples_scaled = alpha * real_samples
        e_target = real_samples_scaled.square().sum(-1)
        e_res = (real_samples_scaled - samples).square().sum(-1)
        sisnr = 10 * torch.log10(e_target / (e_res + 1e-9)).cpu().numpy()

        self.result["mean"] = np.mean(sisnr)
        self.result["std"] = np.std(sisnr)


class SiSPNR(Metric):
    name = "SiSPNR"

    def __init__(self, sampling_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr = sampling_rate
        self.hop_length = int(self.sr / 100)
        self.n_fft = int(2048 / (44100 / self.sr))

    def better(self, first, second):
        return first > second

    @staticmethod
    def get_frobenius_norm(x):
        return torch.linalg.norm(x, dim=(-1, -2))

    @staticmethod
    def get_frobenius_inner_product(x1, x2):
        batched = torch.empty(x1.size(0))
        product = torch.bmm(x1.transpose(-1, -2), x2)
        for i in range(len(batched)):
            batched[i] = torch.trace(product[i])
        return batched

    def _compute(self, samples, real_samples, epoch_num, epoch_info):

        spectrum_hat = get_power_spectrum(samples, n_fft=self.n_fft, hop_length=self.hop_length)  # (B, T, F)
        spectrum = get_power_spectrum(real_samples, n_fft=self.n_fft, hop_length=self.hop_length)  # (B, T, F)

        fro_scalar_product = self.get_frobenius_inner_product(spectrum_hat, spectrum)
        fro_norm = self.get_frobenius_norm(spectrum)

        alpha = fro_scalar_product / (fro_norm ** 2 + 1e-9)
        while len(alpha.shape) < len(spectrum.shape):
            alpha = alpha.unsqueeze(-1)
        spectrum_scaled = alpha * spectrum

        e_target = self.get_frobenius_norm(spectrum_scaled) ** 2
        e_res = self.get_frobenius_norm(spectrum_scaled - spectrum) ** 2
        sispnr = 10 * torch.log10(e_target / (e_res + 1e-9)).cpu().numpy()

        self.result["mean"] = np.mean(sispnr)
        self.result["std"] = np.std(sispnr)


class STOI(Metric):
    name = "STOI"

    def better(self, first, second):
        return first > second

    def __init__(self, sr=16000, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        real_samples = real_samples.squeeze().cpu().numpy()
        samples = samples.squeeze().cpu().numpy()
        if real_samples.ndim == 1:
            real_samples = real_samples[None]
            samples = samples[None]

        stois = []
        for s_real, s_fake in zip(real_samples, samples):
            s = stoi(s_real, s_fake, self.sr, extended=True)
            stois.append(s)
        self.result["mean"] = np.mean(stois)
        self.result["std"] = np.std(stois)


class PESQ(Metric):
    name = "PESQ"

    def better(self, first, second):
        return first > second

    def __init__(self, sampling_rate=16000, **kwargs):
        super().__init__(**kwargs)
        self.sr = sampling_rate

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        samples /= samples.abs().max(-1, keepdim=True)[0]
        real_samples /= real_samples.abs().max(-1, keepdim=True)[0]

        if len(samples.shape) == 3:
            samples = samples.squeeze(1)
            real_samples = real_samples.squeeze(1)

        samples = samples.cpu().numpy()
        real_samples = real_samples.cpu().numpy()

        pesqs = []
        for s_real, s_fake in zip(real_samples, samples):
            try:
                p = pesq(self.sr, s_real, s_fake, mode="wb")
            except:
                p = 1
            pesqs.append(p)

        self.result["mean"] = np.mean(pesqs)
        self.result["std"] = np.std(pesqs)


class MOSNet(Metric):
    name = "MOSNet"

    def __init__(self, weights_path: str, pretrained_path: str, sr=22050, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        if HydraConfig.initialized():
            orig_cwd = hydra.utils.get_original_cwd()
            weights_path = os.path.join(orig_cwd, weights_path)
            pretrained_path = os.path.join(orig_cwd, pretrained_path)

        self.mos_net = Wav2Vec2MOS(weights_path, pretrained_path, device=device)
        self.sr = sr
        self.device = device

    def better(self, first, second):
        return first > second

    def _compute_per_split(self, split):
        return self.mos_net.calculate(split)

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        required_sr = self.mos_net.sample_rate
        resample = torchaudio.transforms.Resample(
            orig_freq=self.sr, new_freq=required_sr
        ).to(samples.device)

        samples /= samples.abs().max(-1, keepdim=True)[0]
        samples = [resample(s).squeeze() for s in samples]

        splits = [
            samples[i:i + self.num_splits]
            for i in range(0, len(samples), self.num_splits)
        ]
        fid_per_splits = [self._compute_per_split(split) for split in splits]
        self.result["mean"] = np.mean(fid_per_splits)
        self.result["std"] = np.std(fid_per_splits)
