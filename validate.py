import hydra
import torch
import numpy as np
import itertools
from diffusion import Diffusion
import torchaudio
import os

import metrics

from tqdm import tqdm

def calculate_all_metrics(wavs, reference_wavs, metrics, n_max_files=None):
    scores = {metric.name: [] for metric in metrics}
    for x, y in tqdm(
        itertools.islice(zip(wavs, reference_wavs), n_max_files),
        total=n_max_files if n_max_files is not None else len(wavs),
        desc="Calculating metrics",
    ):
        try:
            #x = librosa.util.normalize(x[: min(len(x), len(y))])
            #y = librosa.util.normalize(y[: min(len(x), len(y))])
            x = torch.from_numpy(x)[None, None]
            y = torch.from_numpy(y)[None, None]
            for metric in metrics:
                metric._compute(x, y, None, None)
                scores[metric.name] += [metric.result["mean"]]
        except Exception:
           pass
    scores = {k: (np.mean(v), np.std(v)) for k, v in scores.items()}
    return scores


class ValInferencer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.diffusion = Diffusion(beta_type="triangle")

    @property
    def sampling_rate(self):
        return self.cfg.dataset.val_dataset.sampling_rate

    @property
    def device(self):
        return self.cfg.device

    def setup_datasets(self):
        self.val_dataset = hydra.utils.instantiate(self.cfg.dataset.val_dataset)

    def setup_loaders(self):
        self.val_loader = hydra.utils.instantiate(self.cfg.dataloader.val_dataloader, self.val_dataset)

    def setup_model(self, load_from_checkpoint: bool = False):
        model = hydra.utils.instantiate(self.cfg.model.model)
        if load_from_checkpoint:
            model.load_state_dict(torch.load(self.cfg.checkpoint_path, map_location="cpu"))

        self.model = model.to(self.device)

    def setup_metrics(self):
        self.metrics_list = (
            metrics.MOSNet(weights_path="weights/wave2vec2mos.pth", pretrained_path="weights/fb_w2v2_pretrained", sr=self.sampling_rate, device=self.device),
            metrics.ScaleInvariantSignalToDistortionRatio(),
            metrics.LSD(sampling_rate=self.sampling_rate),
            metrics.STOI(sr=self.sampling_rate),
            metrics.PESQ(sampling_rate=self.sampling_rate),
        )

    def generate_predictions(self):
        self.model.eval()
        fake_samples = []
        real_samples = []
        for i, (x, y) in tqdm(enumerate(self.val_loader)):
            if i < 156:
                continue
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                print(x.shape)
                y_pred = self.diffusion.generate(self.model, x.to(self.device), n_steps=100)
                print(y_pred.shape)
                fake_samples.append(y_pred.view(-1).cpu().numpy())
                real_samples.append(y.view(-1).cpu().numpy())
                if not os.path.exists(f"logs/{self.cfg.experiment_name}/predictions"):
                    os.makedirs(f"logs/{self.cfg.experiment_name}/predictions")
                if not os.path.exists(f"logs/{self.cfg.experiment_name}/original"):
                    os.makedirs(f"logs/{self.cfg.experiment_name}/original")
                if not os.path.exists(f"logs/{self.cfg.experiment_name}/noisy"):
                    os.makedirs(f"logs/{self.cfg.experiment_name}/noisy")
                torchaudio.save(f"logs/{self.cfg.experiment_name}/noisy/audio_{i}.wav", x.view(1, -1).cpu(), self.sampling_rate)    
                torchaudio.save(f"logs/{self.cfg.experiment_name}/predictions/audio_{i}.wav", y_pred.view(1, -1).cpu(), self.sampling_rate)
                torchaudio.save(f"logs/{self.cfg.experiment_name}/original/audio_{i}.wav", y.view(1, -1).cpu(), self.sampling_rate)
                
        return fake_samples, real_samples


    def compute_metrics(self, fake_samples, real_samples, epoch_info, **kwargs):
        scores = calculate_all_metrics(fake_samples, real_samples, self.metrics_list)
        for metric in self.metrics_list:
            epoch_info[f"inference_metrics_mean/{metric.name}"] = scores[metric.name][0]
            epoch_info[f"inference_metrics_std/{metric.name}"] = scores[metric.name][1]
        return fake_samples



@hydra.main(config_path="configs", config_name="main_config.yaml", version_base=None)
def main(cfg):
    inferencer = ValInferencer(cfg)
    inferencer.setup_datasets()
    inferencer.setup_loaders()
    inferencer.setup_model(load_from_checkpoint=True)
    inferencer.setup_metrics()
    f, r = inferencer.generate_predictions()
    info = {}
    inferencer.compute_metrics(f, r, info)
    print(info)
    

if __name__ == "__main__":
    main()