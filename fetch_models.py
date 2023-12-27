import hashlib
import os
import subprocess

import huggingface_hub
import k_diffusion as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from requests.exceptions import HTTPError


def fetch(url_or_path):
    if url_or_path.startswith("http:") or url_or_path.startswith("https:"):
        _, ext = os.path.splitext(os.path.basename(url_or_path))
        cachekey = hashlib.md5(url_or_path.encode("utf-8")).hexdigest()
        cachename = f"{cachekey}{ext}"
        if not os.path.exists(f"cache/{cachename}"):
            os.makedirs("tmp", exist_ok=True)
            os.makedirs("cache", exist_ok=True)
            subprocess.run(["curl", f"{url_or_path}", "-o", f"tmp/{cachename}"])
            os.rename(f"tmp/{cachename}", f"cache/{cachename}")
        return f"cache/{cachename}"
    return url_or_path


class NoiseLevelAndTextConditionedUpscaler(nn.Module):
    def __init__(self, inner_model, sigma_data=1.0, embed_dim=256):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.low_res_noise_embed = K.layers.FourierFeatures(1, embed_dim, std=2)

    def forward(self, input, sigma, low_res, low_res_sigma, c, **kwargs):
        cross_cond, cross_cond_padding, pooler = c
        c_in = 1 / (low_res_sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = low_res_sigma.log1p()[:, None]
        c_in = K.utils.append_dims(c_in, low_res.ndim)
        low_res_noise_embed = self.low_res_noise_embed(c_noise)
        low_res_in = F.interpolate(low_res, scale_factor=2, mode="nearest") * c_in
        mapping_cond = torch.cat([low_res_noise_embed, pooler], dim=1)
        return self.inner_model(
            input,
            sigma,
            unet_cond=low_res_in,
            mapping_cond=mapping_cond,
            cross_cond=cross_cond,
            cross_cond_padding=cross_cond_padding,
            **kwargs,
        )


def make_upscaler_model(
    config_path, model_path, pooler_dim=768, train=False, device="cpu"
):
    config = K.config.load_config(open(config_path))
    model = K.config.make_model(config)
    model = NoiseLevelAndTextConditionedUpscaler(
        model,
        sigma_data=config["model"]["sigma_data"],
        embed_dim=config["model"]["mapping_cond_dim"] - pooler_dim,
    )
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_ema"])
    model = K.config.make_denoiser_wrapper(config)(model)
    if not train:
        model = model.eval().requires_grad_(False)
    return model.to(device)


def download_from_huggingface(repo, filename):
    while True:
        try:
            return huggingface_hub.hf_hub_download(repo, filename)
        except HTTPError as e:
            if e.response.status_code == 401:
                # Need to log into huggingface api
                huggingface_hub.interpreter_login()
                continue
            elif e.response.status_code == 403:
                # Need to do the click through license thing
                print(
                    f"Go here and agree to the click through license on your account: https://huggingface.co/{repo}"
                )
                input("Hit enter when ready:")
                continue
            else:
                raise e


def fetch_models():
    # sd_model_path = download_from_huggingface("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt")
    vae_840k_model_path = download_from_huggingface(
        "stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.ckpt"
    )
    vae_560k_model_path = download_from_huggingface(
        "stabilityai/sd-vae-ft-ema-original", "vae-ft-ema-560000-ema-pruned.ckpt"
    )
    return vae_840k_model_path, vae_560k_model_path
