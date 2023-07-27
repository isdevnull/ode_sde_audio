import hydra
import os
import torch
import numpy as np
import random

from datasets import InfiniteSampler
from diffusion import Diffusion
import matplotlib.pyplot as plt
from dataset_utils import draw_spec

import wandb
from tqdm import tqdm


class AudioDiffusionTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.diffusion = Diffusion(beta_type="triangle")

    @property
    def device(self):
        return self.cfg.device

    def setup_datasets(self):
        self.train_dataset = hydra.utils.instantiate(self.cfg.dataset.train_dataset)
        self.val_dataset = hydra.utils.instantiate(self.cfg.dataset.val_dataset)

    def setup_loaders(self):
        sampler = InfiniteSampler(self.train_dataset, shuffle=True)
        self.train_loader = iter(hydra.utils.instantiate(self.cfg.dataloader.train_dataloader, sampler=sampler))
        self.val_loader = hydra.utils.instantiate(self.cfg.dataloader.val_dataloader)

    def setup_model(self, load_from_checkpoint: bool = False):
        model = hydra.utils.instantiate(self.cfg.model.model)
        if load_from_checkpoint:
            model.load_state_dict(torch.load(self.cfg.checkpoint_path, map_location="cpu"))

        self.model = model.to(self.device)

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def train_epoch(self) -> dict:
        loss_acc = 0.0
        aggregated_loss = []
        self.model.train()
        for i in tqdm(range(1, self.cfg.n_iters_per_epoch * self.cfg.accumulate_every + 1)):
            x, y = next(self.train_loader)
            x, y = x.to(self.device), y.to(self.device)
            loss = self.diffusion(self.model, x0=x, x1=y)
            loss.backward()
            loss_acc += loss.detach().cpu().item() / (self.cfg.log_every_iter * self.cfg.accumulate_every)
            if i % self.cfg.accumulate_every == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.cfg.log_every_iter > self.cfg.n_iters_per_epoch:
                    self.cfg.log_every_iter = self.cfg.n_iters_per_epoch
                if i % (self.cfg.log_every_iter * self.cfg.accumulate_every) == 0:
                    wandb.log({"train/mse_loss_per_acc": loss_acc})
                    aggregated_loss.append(loss_acc)
                    loss_acc = 0.0

        return {"train/mse_loss_per_epoch": sum(aggregated_loss) / len(aggregated_loss)}

    def val_epoch(self) -> dict:
        self.model.eval()
        aggregated_loss = []
        for _, (x, y) in tqdm(enumerate(self.val_loader)):
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                loss = self.diffusion(self.model, x0=x, x1=y)
                aggregated_loss.append(loss.item())
            
        return {"val/mse_loss_per_epoch": sum(aggregated_loss) / len(aggregated_loss)}

    def train(self):
        self.setup_datasets()
        self.setup_loaders()
        self.setup_model()
        self.setup_optimizer()

        for epoch in range(1, self.cfg.n_epochs + 1):
            aggregated_info = {"epoch": epoch}
            train_epoch_info = self.train_epoch()
            aggregated_info.update(train_epoch_info)
            val_epoch_info = self.val_epoch()
            aggregated_info.update(val_epoch_info)
            
            # log train/val epoch losses
            wandb.log(aggregated_info)
            
            # visualize samples
            if epoch % self.cfg.visualize_every_epoch == 0:
                self.model.eval()
                for _, (x, y) in enumerate(self.val_loader):
                    with torch.no_grad():
                        y_pred = self.diffusion.generate(self.model, x.to(self.device), n_steps=100)

                    log_data = {}
                    for idx in range(y_pred.shape[0]):
                        sr = self.cfg.dataset.sampling_rate
                        spec_fig, ax = plt.subplots(3, 1, figsize=(12, 12))
                        draw_spec(x[idx].view(-1).cpu().numpy(), "noisy", sr=sr, mel=False, ax=ax[0])
                        draw_spec(y_pred[idx].view(-1).cpu().numpy(), "pred", sr=sr, mel=False, ax=ax[1])
                        draw_spec(y[idx].view(-1).cpu().numpy(), "orig", sr=sr, mel=False, ax=ax[2])


                        log_data.update({
                            f"test_sample_{idx}/noisy": wandb.Audio(x[idx].view(-1).cpu().numpy(), sample_rate=sr, caption=f"{epoch=}"),
                            f"test_sample_{idx}/pred": wandb.Audio(y_pred[idx].view(-1).cpu().numpy(), sample_rate=sr, caption=f"{epoch=}"),
                            f"test_sample_{idx}/ref": wandb.Audio(y[idx].view(-1).cpu().numpy(), sample_rate=sr, caption=f"{epoch=}"),
                            f"test_sample_{idx}/specs": wandb.Image(spec_fig, caption=f"{epoch=}"),
                        })

                        
                    wandb.log(log_data)
                    break

            # save model checkpoint
            if epoch % self.cfg.save_every_epoch == 0:
                save_dir = os.path.join(self.cfg.checkpoint_dir, self.cfg.experiment_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(self.model.state_dict(), os.path.join(save_dir, f"ckpt_{epoch=}.pth"))


@hydra.main("configs", "main_config", version_base=None)
def main(cfg):
    setup_seed(cfg.seed, False)
    
    wandb.init(project="audio_flows", name=cfg.experiment_name)
    trainer = AudioDiffusionTrainer(cfg)
    trainer.train()


def setup_seed(seed, cudnn_benchmark_off):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_benchmark_off:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    main()