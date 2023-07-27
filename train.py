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

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration


class AudioDiffusionTrainer:
    def __init__(self, cfg, accelerator: Accelerator):
        self.cfg = cfg
        self.diffusion = Diffusion(beta_type="triangle")
        self.accelerator = accelerator

    @property
    def device(self):
        return self.accelerator.device

    def setup_datasets(self):
        self.train_dataset = hydra.utils.instantiate(self.cfg.dataset.train_dataset)
        self.val_dataset = hydra.utils.instantiate(self.cfg.dataset.val_dataset)

    def setup_loaders(self):
        sampler = InfiniteSampler(self.train_dataset, shuffle=True)
        self.train_loader = iter(hydra.utils.instantiate(self.cfg.dataloader.train_dataloader, self.train_dataset, sampler=sampler))
        self.val_loader = hydra.utils.instantiate(self.cfg.dataloader.val_dataloader, self.val_dataset)

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
            with self.accelerator.accumulate(self.model):
                x, y = next(self.train_loader)
                #x, y = x.to(self.device), y.to(self.device)
                loss = self.diffusion(self.model, x0=x, x1=y)
                self.accelerator.backward(loss)
                loss_acc += loss.detach().cpu().item() / (self.cfg.log_every_iter * self.cfg.accumulate_every)

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.cfg.log_every_iter > self.cfg.n_iters_per_epoch:
                    self.cfg.log_every_iter = self.cfg.n_iters_per_epoch
                if i % (self.cfg.log_every_iter * self.cfg.accumulate_every) == 0:
                    self.accelerator.log({"train/mse_loss_per_acc": loss_acc})
                    aggregated_loss.append(loss_acc)
                    loss_acc = 0.0

        return {"train/mse_loss_per_epoch": sum(aggregated_loss) / len(aggregated_loss)}

    def val_epoch(self) -> dict:
        self.model.eval()
        aggregated_loss = []
        for i, (x, y) in tqdm(enumerate(self.val_loader)):
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                loss = self.diffusion(self.model, x0=x, x1=y)
                aggregated_loss.append(loss.item())
            
            # hardcoded for now, lack of time
            if i == 800:
                break
            
        return {"val/mse_loss_per_epoch": sum(aggregated_loss) / len(aggregated_loss)}

    def train(self):
        self.setup_datasets()
        self.setup_loaders()
        self.setup_model()
        self.setup_optimizer()

        self.accelerator.prepare(self.model, self.optimizer, self.train_loader, self.val_loader)
        if self.cfg.full_state_ckpt is not None:
            self.accelerator.load_state(self.cfg.full_state_ckpt)

        for epoch in range(1, self.cfg.n_epochs + 1):
            aggregated_info = {"epoch": epoch}
            train_epoch_info = self.train_epoch()
            aggregated_info.update(train_epoch_info)
            val_epoch_info = self.val_epoch()
            aggregated_info.update(val_epoch_info)
            
            # log train/val epoch losses
            self.accelerator.log(aggregated_info)
            
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

                    wandb_tracker = self.accelerator.get_tracker("wandb", unwrap=True)
                    with self.accelerator.on_main_process:
                        wandb_tracker.log(log_data)
                 
                    break

            # save model checkpoint
            if epoch % self.cfg.save_model_every_epoch == 0:

                self.accelerator.wait_for_everyone()
                save_dir = os.path.join(self.accelerator.project_configuration.project_dir, "model_ckpt")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                self.accelerator.save(self.accelerator.unwrap_model(self.model).state_dict(), os.path.join(save_dir, f"ckpt_{epoch=}.pth"))
            
            # save full state
            if epoch % self.cfg.save_full_state_every == 0:
                full_state_path = os.path.join(self.accelerator.project_configuration.project_dir, "full_state_ckpt")
                if not os.path.exists(full_state_path):
                    os.makedirs(full_state_path)
                self.accelerator.save_state(full_state_path)


@hydra.main("configs", "main_config", version_base=None)
def main(cfg):
    
    # setup project paths
    project_cfg = ProjectConfiguration(
        project_dir=f"experiments/{cfg.experiment_name}",
        automatic_checkpoint_naming=True,
        total_limit=1,
    )

    # initialize accelerator
    accelerator = Accelerator(
        project_config=project_cfg,
        log_with="wandb",
        gradient_accumulation_steps=cfg.accumulate_every
    )

    # just test what it gives
    cfg_log = dict(
        n_epochs=cfg.n_epochs,
        n_iters_per_epoch=cfg.n_iters_per_epoc,
        train_batch_size=cfg.train_batch_size,
        val_batch_size=cfg.val_batch_size,
        accumulate_every=cfg.accumulate_every,
        save_every_epoch=cfg.save_every_epoch,
        log_every_iter=cfg.log_every_iter,
        compute_metric_every_epoch=cfg.compute_metric_every_epoch, # not working now
        visualize_every_epoch=cfg.visualize_every_epoch,
        lr=2e-4
    )
    
    # initalize specified experiment trackers
    accelerator.init_trackers(project_name="audio_flows", config=cfg_log)

    # intialize trainer and pass accelerator to it
    trainer = AudioDiffusionTrainer(cfg, accelerator)
    
    # train
    trainer.train()

    # end training
    accelerator.end_training()


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