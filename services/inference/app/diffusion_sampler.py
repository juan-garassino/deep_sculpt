import math
from typing import Optional

import torch


class NoiseScheduler:
    def __init__(
        self,
        schedule_type: str = "cosine",
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.schedule_type = schedule_type
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.betas = self._create_beta_schedule().to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _create_beta_schedule(self) -> torch.Tensor:
        if self.schedule_type == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        if self.schedule_type == "cosine":
            return self._cosine_beta_schedule()
        if self.schedule_type == "sigmoid":
            betas = torch.linspace(-6, 6, self.timesteps)
            betas = torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
            return betas
        raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def _cosine_beta_schedule(self) -> torch.Tensor:
        s = 0.008
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)


def sample_diffusion(
    model: torch.jit.ScriptModule,
    scheduler: NoiseScheduler,
    shape: tuple[int, ...],
    device: str,
    num_inference_steps: int = 50,
    conditioning: Optional[torch.Tensor] = None,
    class_labels: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    x = torch.randn(shape, device=device)
    timesteps = torch.linspace(
        scheduler.timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device
    )

    for t in timesteps:
        t_batch = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)
        try:
            if conditioning is not None and guidance_scale != 1.0:
                eps_uncond = model(x, t_batch)
                eps_cond = model(x, t_batch, conditioning) if class_labels is None else model(
                    x, t_batch, conditioning, class_labels
                )
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            elif conditioning is not None and class_labels is not None:
                eps = model(x, t_batch, conditioning, class_labels)
            elif conditioning is not None:
                eps = model(x, t_batch, conditioning)
            elif class_labels is not None:
                eps = model(x, t_batch, None, class_labels)
            else:
                eps = model(x, t_batch)
        except Exception:
            eps = model(x, t_batch)

        alpha = scheduler.alphas[t]
        alpha_cumprod = scheduler.alphas_cumprod[t]
        beta = scheduler.betas[t]

        coef = 1.0 / torch.sqrt(alpha)
        coef_eps = beta / torch.sqrt(1.0 - alpha_cumprod)
        x = coef * (x - coef_eps * eps)

        if t > 0:
            noise = torch.randn_like(x)
            var = scheduler.posterior_variance[t]
            x = x + torch.sqrt(var) * noise

    return x
