import argparse
import subprocess
import sys

sys.path.extend(["./taming-transformers", "./stable-diffusion", "./latent-diffusion"])

import time

import k_diffusion as K
import numpy as np
import torch
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.transforms import functional as TF

from fetch_models import download_from_huggingface, fetch, make_upscaler_model
from nn_module import CFGUpscaler, CLIPEmbedder, CLIPTokenizerTransform


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    cpu = torch.device("cpu")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model = model.to(cpu).eval().requires_grad_(False)
    return model


class SDUpscaler:
    def __init__(self, args) -> None:
        self.num_samples = 1
        self.batch_size = 1
        self.decoder = "finetuned_840k"
        self.guidance_scale = 1
        self.noise_aug_level = 0
        self.noise_aug_type = "gaussian"
        self.sampler = "k_dpm_adaptive"
        self.steps = 50
        self.tol_scale = 0.25
        self.eta = 1.0
        self.device = torch.device("cuda")

        self.SD_C = 4  # Latent dimension
        self.SD_F = 8  # Latent patch size (pixels per latent)
        self.SD_Q = (
            0.18215  # sd_model.scale_factor; scaling for latents in first stage models
        )

        model_up = make_upscaler_model(
            fetch(
                "https://models.rivershavewings.workers.dev/config_laion_text_cond_latent_upscaler_2.json"
            ),
            fetch(
                "https://models.rivershavewings.workers.dev/laion_text_cond_latent_upscaler_2_1_00470000_slim.pth"
            ),
        )
        vae_840k_model_path = download_from_huggingface(
            "stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.ckpt"
        )

        vae_560k_model_path = download_from_huggingface(
            "stabilityai/sd-vae-ft-ema-original", "vae-ft-ema-560000-ema-pruned.ckpt"
        )
        sd_model_path = download_from_huggingface(
            "CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt"
        )

        device = torch.device("cuda")

        vae_model_840k = load_model_from_config(
            "latent-diffusion/models/first_stage_models/kl-f8/config.yaml",
            vae_840k_model_path,
        )
        vae_model_560k = load_model_from_config(
            "latent-diffusion/models/first_stage_models/kl-f8/config.yaml",
            vae_560k_model_path,
        )

        self.vae_model_840k = vae_model_840k.to(device)
        self.vae_model_560k = vae_model_560k.to(device)
        self.model_up = model_up.to(device)

        self.timestamp = int(time.time())
        if args.seed is not None:
            seed_everything(args.seed)
        else:
            seed_everything(self.timestamp)

        self.input_image = Image.open(args.input_image).convert("RGB")
        subprocess.run(
            [
                "python",
                "./stable-diffusion/scripts/txt2img.py",
                "--prompt",
                f"{args.prompt}",
                "--plms",
                "--ckpt",
                f"{sd_model_path}",
                "--skip_grid",
                "--n_samples",
                f"{self.num_samples}",
            ]
        )

    def run(self):
        tok_up = CLIPTokenizerTransform()
        text_encoder_up = CLIPEmbedder(device=self.device)

        @torch.no_grad()
        def condition_up(prompts):
            return text_encoder_up(tok_up(prompts))

        uc = condition_up(self.batch_size * [""])
        c = condition_up(self.batch_size * [self.prompt])

        if self.decoder == "finetuned_840k":
            vae = self.vae_model_840k
        elif self.decoder == "finetuned_560k":
            vae = self.vae_model_560k

        image = self.input_image
        image = TF.to_tensor(image).to(self.device) * 2 - 1
        low_res_latent = vae.encode(image.unsqueeze(0)).sample() * self.SD_Q
        low_res_decoded = vae.decode(low_res_latent / self.SD_Q)

        [_, C, H, W] = low_res_latent.shape

        # Noise levels from stable diffusion.
        sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512

        model_wrap = CFGUpscaler(self.model_up, uc, cond_scale=self.guidance_scale)
        low_res_sigma = torch.full(
            [self.batch_size], self.noise_aug_level, device=self.device
        )
        x_shape = [self.batch_size, C, 2 * H, 2 * W]

        def do_sample(noise, extra_args):
            # We take log-linear steps in noise-level from sigma_max to sigma_min, using one of the k diffusion samplers.
            sigmas = (
                torch.linspace(np.log(sigma_max), np.log(sigma_min), self.steps + 1)
                .exp()
                .to(self.device)
            )
            if self.sampler == "k_euler":
                return K.sampling.sample_euler(
                    model_wrap, noise * sigma_max, sigmas, extra_args=extra_args
                )
            elif self.sampler == "k_euler_ancestral":
                return K.sampling.sample_euler_ancestral(
                    model_wrap,
                    noise * sigma_max,
                    sigmas,
                    extra_args=extra_args,
                    eta=self.eta,
                )
            elif self.sampler == "k_dpm_2_ancestral":
                return K.sampling.sample_dpm_2_ancestral(
                    model_wrap,
                    noise * sigma_max,
                    sigmas,
                    extra_args=extra_args,
                    eta=self.eta,
                )
            elif self.sampler == "k_dpm_fast":
                return K.sampling.sample_dpm_fast(
                    model_wrap,
                    noise * sigma_max,
                    sigma_min,
                    sigma_max,
                    self.steps,
                    extra_args=extra_args,
                    eta=self.eta,
                )
            elif self.sampler == "k_dpm_adaptive":
                sampler_opts = dict(
                    s_noise=1.0,
                    rtol=self.tol_scale * 0.05,
                    atol=self.tol_scale / 127.5,
                    pcoeff=0.2,
                    icoeff=0.4,
                    dcoeff=0,
                )
                return K.sampling.sample_dpm_adaptive(
                    model_wrap,
                    noise * sigma_max,
                    sigma_min,
                    sigma_max,
                    extra_args=extra_args,
                    eta=self.eta,
                    **sampler_opts,
                )

        image_id = 0
        for _ in range((self.num_samples - 1) // self.batch_size + 1):
            if self.noise_aug_type == "gaussian":
                latent_noised = (
                    low_res_latent
                    + self.noise_aug_level * torch.randn_like(low_res_latent)
                )
            elif self.noise_aug_type == "fake":
                latent_noised = low_res_latent * (self.noise_aug_level**2 + 1) ** 0.5
            extra_args = {
                "low_res": latent_noised,
                "low_res_sigma": low_res_sigma,
                "c": c,
            }
            noise = torch.randn(x_shape, device=self.device)
            up_latents = do_sample(noise, extra_args)

            pixels = vae.decode(
                up_latents / self.SD_Q
            )  # equivalent to sd_model.decode_first_stage(up_latents)
            pixels = pixels.add(1).div(2).clamp(0, 1)

            # Display and save samples.
            for j in range(pixels.shape[0]):
                img = TF.to_pil_image(pixels[j])
                img.save("output.png")
                image_id += 1


parser = argparse.ArgumentParser()
parser.add_argument("prompt", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
upscaler = SDUpscaler(parser.parse_args())
upscaler.run()
