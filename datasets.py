import random
import os
from os import path
import torch
import numpy as np
import librosa

from librosa.util import normalize
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch
import numpy as np
import random

from scipy.signal import sosfiltfilt
from scipy.signal import cheby1
from scipy.signal import resample_poly


def split_audios(audios, segment_size, split):
    audios = [torch.FloatTensor(audio).unsqueeze(0) for audio in audios]
    if split:
        if audios[0].size(1) >= segment_size:
            max_audio_start = audios[0].size(1) - segment_size
            audio_start = random.randint(0, max_audio_start)
            audios = [
                audio[:, audio_start : audio_start + segment_size] for audio in audios
            ]
        else:
            audios = [
                torch.nn.functional.pad(
                    audio,
                    (0, segment_size - audio.size(1)),
                    "constant",
                )
                for audio in audios
            ]
    audios = [audio.squeeze(0).numpy() for audio in audios]
    return audios


class InfiniteSampler(torch.utils.data.Sampler):
    """
        Taken from Nvidia EDM repository
    """
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1



class VoicebankDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        noisy_wavs_dir,
        clean_wavs_dir=None,
        path_prefix=None,
        segment_size=8192,
        sampling_rate=16000,
        split=True,
        shuffle=False,
        device=None,
        input_freq=None,
    ):
        if path_prefix:
            if clean_wavs_dir:
                clean_wavs_dir = os.path.join(path_prefix, clean_wavs_dir)
            noisy_wavs_dir = os.path.join(path_prefix, noisy_wavs_dir)

        if clean_wavs_dir:
            self.audio_files = self.read_files_list(clean_wavs_dir, noisy_wavs_dir)
        else:
            self.audio_files = self.read_noisy_list(noisy_wavs_dir)

        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)

        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.device = device
        self.input_freq = input_freq

    @staticmethod
    def read_files_list(clean_wavs_dir, noisy_wavs_dir):
        fn_lst_clean = os.listdir(clean_wavs_dir)
        fn_lst_noisy = os.listdir(noisy_wavs_dir)
        assert set(fn_lst_clean) == set(fn_lst_noisy)
        return sorted(fn_lst_clean)

    @staticmethod
    def read_noisy_list(noisy_wavs_dir):
        fn_lst_noisy = os.listdir(noisy_wavs_dir)
        return sorted(fn_lst_noisy)

    def make_input(
        self, clean_audio: np.ndarray, noisy_audio: np.ndarray
    ) -> np.ndarray:
        """
        Input arguments have the same length
        """
        raise NotImplementedError()

    def split_audios(self, audios):
        return split_audios(audios, self.segment_size, self.split)

    def __getitem__(self, index):
        fn = self.audio_files[index]

        if self.clean_wavs_dir:
            clean_audio = librosa.load(
                os.path.join(self.clean_wavs_dir, fn),
                sr=self.sampling_rate,
                res_type="polyphase",
            )[0]
        else:
            clean_audio = None

        noisy_audio = librosa.load(
            os.path.join(self.noisy_wavs_dir, fn),
            sr=self.sampling_rate,
            res_type="polyphase",
        )[0]

        if clean_audio is not None:
            clean_audio, noisy_audio = self.split_audios([clean_audio, noisy_audio])
        else:
            noisy_audio = self.split_audios([noisy_audio])[0]

        input_audio = self.make_input(clean_audio, noisy_audio)

        if clean_audio is not None:
            assert input_audio.shape[1] == clean_audio.size
            audio = torch.FloatTensor(normalize(clean_audio) * 0.95)
            audio = audio.unsqueeze(0)
        else:
            audio = torch.Tensor()

        input_audio = torch.FloatTensor(input_audio)

        return input_audio, audio

    def __len__(self):
        return len(self.audio_files)


class Voicebank1ChannelDataset(VoicebankDataset):
    def make_input(self, clean_audio, noisy_audio):
        return normalize(noisy_audio)[None] * 0.95


def collate_fn_vctk_bwe(batch):
        wav_list = list()
        wav_l_list = list()
        band_list = list()
        for wav, wav_l, band in batch:
            wav_list.append(wav)
            wav_l_list.append(wav_l)
            band_list.append(band)
        wav_list = torch.stack(wav_list, dim=0).squeeze(1)
        wav_l_list = torch.stack(wav_l_list, dim=0).squeeze(1)
        band_list = torch.stack(band_list, dim=0)

        return wav_list, wav_l_list, band_list


# def create_vctk_dataloader(hparams, cv, sr=24000):
#     if cv == 0:
#         dataset =VCTKMultiSpkDataset(hparams, cv)
#         sampler = InfiniteSampler(dataset)
#         return DataLoader(dataset=dataset,
#                           batch_size=hparams.train.batch_size,
#                           num_workers=8,
#                           collate_fn=collate_fn_vctk_bwe,
#                           pin_memory=True,
#                           drop_last=True,
#                           sampler=sampler,
#                           prefetch_factor=4)
#     else:
#         return DataLoader(dataset=VCTKMultiSpkDataset(hparams, cv) if cv == 1 else VCTKMultiSpkDataset(hparams, cv, sr),
#                           collate_fn=collate_fn_vctk_bwe,
#                           batch_size=hparams.train.batch_size if cv == 1 else 1,
#                           drop_last=True if cv == 1 else False,
#                           shuffle=False,
#                           num_workers=8,
#                           prefetch_factor=4)


class VCTKMultiSpkDataset(Dataset):
    def __init__(self, hparams, cv=0, sr=24000):  # cv 0: train, 1: val, 2: test
        def _get_datalist(folder, file_format, spk_list, cv):
            _dl = []
            len_spk_list = len(spk_list)
            s = 0
            print(f'full speakers {len_spk_list}')
            for i, spk in enumerate(spk_list):
                if cv == 0:
                    if not (i < int(len_spk_list * self.cv_ratio[0])): continue
                elif cv == 1:
                    if not (int(len_spk_list * self.cv_ratio[0]) <= i and
                            i <= int(len_spk_list * (self.cv_ratio[0] + self.cv_ratio[1]))):
                        continue
                else:
                    if not (int(len_spk_list * self.cv_ratio[0]) <= i and
                            i <= int(len_spk_list * (self.cv_ratio[0] + self.cv_ratio[1]))):
                        continue
                _full_spk_dl = sorted(glob(path.join(spk, file_format)))
                _len = len(_full_spk_dl)
                if (_len == 0): continue
                s += 1
                _dl.extend(_full_spk_dl)

            print(cv, s)
            return _dl

        def _get_spk(folder):
            return sorted(glob(path.join(folder, '*')))  # [1:])

        self.hparams = hparams
        self.cv = cv
        self.cv_ratio = eval(hparams.data.cv_ratio)
        self.sr = sr
        self.directory = hparams.data.dir
        self.dataformat = hparams.data.format

        self.data_list = _get_datalist(self.directory, self.dataformat,
                                       _get_spk(self.directory), self.cv)

        assert len(self.data_list) != 0, "no data found"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        wav, _ = librosa.load(self.data_list[index], sr=self.hparams.audio.sampling_rate)
        wav /= np.max(np.abs(wav))

        if wav.shape[0] < self.hparams.audio.length:
            padl = self.hparams.audio.length - wav.shape[0]
            r = random.randint(0, padl) if self.cv < 2 else padl // 2
            wav = np.pad(wav, (r, padl - r), 'constant', constant_values=0)
        else:
            start = random.randint(0, wav.shape[0] - self.hparams.audio.length)
            wav = wav[start:start + self.hparams.audio.length] if self.cv < 2 \
                else wav[:len(wav) - len(wav) % self.hparams.audio.hop_length]
        wav *= random.random() / 2 + 0.5 if self.cv < 2 else 1

        if self.cv == 0:
            order = random.randint(1, 11)
            ripple = random.choice([1e-9, 1e-6, 1e-3, 1, 5])
            highcut = random.randint(self.hparams.audio.sr_min // 2, self.hparams.audio.sr_max // 2)
        else:
            order = 8
            ripple = 0.05
            if self.cv == 1:
                highcut = random.choice([8000 // 2, 12000 // 2, 16000 // 2, 24000 // 2])
            elif self.cv == 2:
                highcut = self.sr // 2

        nyq = 0.5 * self.hparams.audio.sampling_rate
        hi = highcut / nyq

        if hi == 1:
            wav_l = wav
        else:
            sos = cheby1(order, ripple, hi, btype='lowpass', output='sos')
            wav_l = sosfiltfilt(sos, wav)

            # downsample to the low sampling rate
            wav_l = resample_poly(wav_l, highcut * 2, self.hparams.audio.sampling_rate)
            # upsample to the original sampling rate
            wav_l = resample_poly(wav_l, self.hparams.audio.sampling_rate, highcut * 2)

        if len(wav_l) < len(wav):
            wav_l = np.pad(wav, (0, len(wav) - len(wav_l)), 'constant', constant_values=0)
        elif len(wav_l) > len(wav):
            wav_l = wav_l[:len(wav)]

        # fft_size = self.hparams.audio.filter_length // 2 + 1
        # band = torch.zeros(fft_size, dtype = torch.int64)
        # band[:int(hi * fft_size)] = 1
        return torch.from_numpy(wav).float(), torch.from_numpy(wav_l.copy()).float()