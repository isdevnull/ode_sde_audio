import torch
import torch.nn.functional

from torchsde import sdeint

from typing import Callable
from functools import partial

from torchdiffeq import odeint


def triangle_func(t: torch.tensor, a: float = 1e-9, b: float = 1.3e-4, *args, **kwargs) -> torch.tensor:
    """
        defines triangle function with min_beta(0) = a, min_beta(1) = a, max_beta(0.5) = b
        correctly works on t in [0, 1]
    """
    beta = torch.empty_like(t, device=t.device)
    beta[t <= 0.5] = a + 2 * t[t <= 0.5] * (b - a)
    beta[t > 0.5] = 2 * (1 - t[t > 0.5]) * b + (2 * t[t > 0.5] - 1) * a
    return beta


def constant_func(t: torch.Tensor, epsilon: float = 1e-4, *args, **kwargs) -> torch.tensor:
    return torch.empty_like(t, device=t.device).fill_(epsilon)


def get_beta_function(type: str = "triangle", *args, **kwargs) -> Callable:
    if type == "triangle":
        beta_func = partial(triangle_func, *args, **kwargs)
    else:
        beta_func = partial(constant_func, *args, **kwargs)
    return beta_func


def solve_adaptive_ode(
    model,
    x0,
    n_steps=100,
    begin=0.0,
    end=1.0,
    ode_method="euler",
) -> torch.Tensor:
    
    @torch.no_grad()
    def drift(t, x):
        sigma = torch.ones(x0.shape[0], device=x.device) * t
        return model(x, sigma)

    out = odeint(
        func=drift,
        y0=x0,
        t=torch.linspace(begin, end, n_steps + 1, device=x0.device),
        method=ode_method,
        # options={'step_size': abs(end - begin) / n_steps}
    )
    return out[-1]  # type: ignore


class Diffusion:
    def __init__(self, beta_type: str, beta_min: float = 1e-9, beta_max: float = 1.3e-4, spectral: bool = False):
        self.beta_type = beta_type
        self.beta_min, self.beta_max = beta_min, beta_max
        self.spectral = spectral
        if self.spectral:
            self.window = torch.hann_window(512)
        self.beta_func = get_beta_function(beta_type, a=beta_min, b=beta_max, epsilon=beta_max)

    def get_variance(self, t: torch.Tensor):
        sigma1 = torch.empty_like(t, device=t.device) # [0, t]
        sigma2 = torch.empty_like(t, device=t.device) # [t, 1]
        if self.beta_type == "triangle":
            # because beta is piecewise, define separately for each segment [0, 0.5] and [0.5, 1.0]
            sigma1[t <= 0.5] = (t[t <= 0.5] ** 2) * (self.beta_max - self.beta_min) + self.beta_min * t[t <= 0.5]
            sigma1[t > 0.5] = (self.beta_max - self.beta_min) * (2 * t[t > 0.5]  - t[t > 0.5] ** 2 - 0.75)
            sigma2[t <= 0.5] = self.beta_min * (1 - t[t <= 0.5]) + (1 - t[t <= 0.5] ** 2) * (self.beta_max - self.beta_min)
            sigma2[t > 0.5] = (self.beta_max - self.beta_min) * (t[t > 0.5] - 1) ** 2
        else: # assume constant
            sigma1 = self.beta_max * t
            sigma2 = self.beta_max * (1 - t)
        
        return sigma1, sigma2
    
    def get_interpolant(self, t, x0, x1):
        sigma1, sigma2 = self.get_variance(t)
        s = sigma1 + sigma2
        mu = x0 * sigma2 / s + x1 * sigma1  / s
        cov = sigma1 * sigma2 / s

        x_t = mu + cov * torch.randn_like(x0)
        return x_t, sigma2
    
    def get_vanilla_interpolant(self, t, x0, x1):
        x_t = (1 - t) * x0 + t * x1
        #print(x_t.abs().max(-1, keepdims=True).values.shape, x_t.shape)

        #x_t /= x_t.abs().max(-1, keepdim=True).values

        #x_t *= ((torch.rand(1).to(x0.device) / 2) + 0.5)
        return x_t, None
    
    def generate(self, net, x0, n_steps: int = 100):
        if self.spectral:
            x0 = torch.stft(x0.squeeze(1), n_fft=512, win_length=512, hop_length=128, window=self.window.to(x0.device), center=True, normalized=True, return_complex=True)
            x0 = torch.view_as_real(x0).permute(0, 3, 1, 2)

            pred_spectral = solve_adaptive_ode(net, x0, n_steps=n_steps)
            #pred_spectral = torch.view_as_complex(pred_spectral.permute(0, 2, 3, 1).contiguous())
            #pred = torch.istft(pred_spectral, n_fft=512, win_length=512, hop_length=128, window=self.window.to(x0.device), center=True, normalized=True)
            pred = self.get_istft(pred_spectral)
            pred = pred.unsqueeze(1)
        else:
            pred = solve_adaptive_ode(net, x0, n_steps=n_steps)

        return pred
    
    # def generate_spectral(self, net, x0, n_steps: int = 100):
    #     t = torch.linspace(0., 1., steps=n_steps + 1).to(x0.device)

    #     class SDEWrapper(torch.nn.Module):
    #         """
    #             SDE model wrapper for torchsde.sdeint
    #         """
    #         def __init__(self, model: torch.nn.Module, beta_f: Callable, shape, sde_type: str = "ito", noise_type: str = "general"):
    #             super().__init__()
    #             self.sde_type = sde_type
    #             self.noise_type = noise_type
    #             self.model = model
    #             self.model.eval()
    #             self.beta_f = beta_f
    #             self.shape = shape
            
    #         def f(self, t, y):
    #             with torch.no_grad():
    #                 return self.model(y.view(*self.shape), t).squeeze()

    #         def g(self, t, y):
    #             return torch.sqrt(self.beta_f(t.view(1, 1, 1).expand(y.size(0), y.size(-1), 1)))
        
    #     x0 = torch.stft(x0.squeeze(), n_fft=512, win_length=512, hop_length=128, window=self.window.to(x0.device), center=True, normalized=True, return_complex=True)
    #     x0 = torch.view_as_real(x0).permute(0, 3, 1, 2)
    #     shape = x0.shape
    #     sde_model = SDEWrapper(net, self.beta_func, shape)
        
    #     return sdeint(sde_model, x0.reshape(shape[0], -1), t, dt=1e-2, dt_min=1e-3)[-1]


    # def generate(self, net, x0, n_steps: int = 100):
    #     t = torch.linspace(0., 1., steps=n_steps + 1).to(x0.device)

    #     class SDEWrapper(torch.nn.Module):
    #         """
    #             SDE model wrapper for torchsde.sdeint
    #         """
    #         def __init__(self, model: torch.nn.Module, beta_f: Callable, sde_type: str = "ito", noise_type: str = "general"):
    #             super().__init__()
    #             self.sde_type = sde_type
    #             self.noise_type = noise_type
    #             self.model = model
    #             self.model.eval()
    #             self.beta_f = beta_f
            
    #         def f(self, t, y):
    #             with torch.no_grad():
    #                 return self.model(y.unsqueeze(1), t).squeeze()

    #         def g(self, t, y):
    #             return torch.sqrt(self.beta_f(t.view(1, 1, 1).expand(y.size(0), y.size(-1), 1)))

    #     sde_model = SDEWrapper(net, self.beta_func)
    #     return sdeint(sde_model, x0, t, dt=1e-2, dt_min=1e-3)[-1]

    def get_stft(self, x):
        x = torch.stft(x.squeeze(1), n_fft=512, win_length=512, hop_length=128, window=self.window.to(x.device), center=True, normalized=True, return_complex=True)
        x = torch.view_as_real(x).permute(0, 3, 1, 2)
        return x
    
    def get_istft(self, x):
        assert(x.dim() == 4)
        x = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())
        x = torch.istft(x, n_fft=512, win_length=512, hop_length=128, window=self.window.to(x.device), center=True, normalized=True)
        return x


    def __call__(self, net, x0, x1):
        """
            loss function
        """
        # if self.spectral:
        #     x0 = self.get_stft(x0)
        #     x1 = self.get_stft(x1)

        t = torch.rand(x0.size(0)).to(x0.device)
        while t.dim() < x0.dim():
            t = t[:, None]
        #beta_schedule = self.beta_func(t)
        x_t, sigma2 = self.get_vanilla_interpolant(t, x0, x1)
        if self.spectral:
            x_t = self.get_stft(x_t)
        #x_t /= x_t.abs().max(-1).values

        true_vf = x1 - x0
        true_vf /= true_vf.abs().max(-1, keepdim=True).values
        #true_vf *= ((torch.rand(1, device=x0.device) / 2) + 0.5)

        if self.spectral:
            true_vf = self.get_stft(true_vf)

        pred_vf = net(x_t, t.view(-1))
        #true_vf = beta_schedule / sigma2 * (x1 - x_t)
        
        #return torch.nn.functional.l1_loss(pred_vf, true_vf, reduction="none").sum() / x0.size(0)
        return (pred_vf - true_vf).square().sum() / x0.size(0)



if __name__ == "__main__":
    from omegaconf import OmegaConf
    from diffwave import DiffWave
    from models.networks import STFTUnet

    device = "cuda:0"

    params = OmegaConf.create(
        dict(
            residual_channels=512,
            residual_layers=27,
            n_mels=80,
            unconditional=True,
            dilation_cycle_length=12,
        )
    )

    spectral = False

    #model = DiffWave(params).to(device)
    model = STFTUnet(257, 2, spectral=spectral).to(device)
    print(sum(p.numel() for p in model.parameters()))

    #y = model(torch.randn(2, 32768).to(device), torch.rand(2).to(device))
    #print(y.shape)

    x = torch.randn(2, 1, 32768).to(device)
    y = torch.randn(2, 1, 32768).to(device)


    diffusion = Diffusion("triangle", spectral=spectral)
    loss = diffusion(model, x0=y, x1=x)
    print(loss)

    s = diffusion.generate(model, torch.randn(2, 1, 32768).to(device))
    print(s.shape)