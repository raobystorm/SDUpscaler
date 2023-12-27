pip install -r requirements.txt
pip install -U git+https://github.com/huggingface/huggingface_hub

git clone https://github.com/CompVis/stable-diffusion
git clone https://github.com/CompVis/taming-transformers
git clone https://github.com/CompVis/latent-diffusion

pip install -e ./taming-transformers
pip install -e ./stable-diffusion
pip install -e ./latent-diffusion
