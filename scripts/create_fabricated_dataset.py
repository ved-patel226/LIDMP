import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

torch.set_float32_matmul_precision("high")

from src.dataset import CatDogDataModule
from src.models.compression.vq_vae import VQVAE
from callbacks.reconstruction_callback import ReconstructionCallback
from torchvision.utils import save_image
import os
from tqdm import tqdm


model = VQVAE(
    h_dim=256,
    n_embeddings=1023,
    embedding_dim=64,
    beta=0.25,
    lr=1e-3,
    n_heads=4,
    n_layers=3,
    patch_size=1,
    load_params=["encoder", "decoder", "vector_quantization"],
    load_path="checkpoints/stage1/vq_vae_quantizer_pretrain-v18.ckpt",
    is_fsq=False,
    num_quantizers=2,
)


data_module = CatDogDataModule(
    data_dir="./data/PetImages",
    batch_size=1,
)
data_module = data_module.setup()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

output_dir = "./data/ReconstructedPetImages"
os.makedirs(output_dir, exist_ok=True)

for i, batch in enumerate(
    tqdm(data_module.train_dataloader(), desc="Reconstructing Images")
):
    images, labels = batch[0].to(device), batch[1]

    _, x_hat, indicies, _ = model(images)

    label = labels.item() if labels.numel() == 1 else labels[0].item()
    subfolder = "Dog" if label == 1 else "Cat"
    subfolder_path = os.path.join(output_dir, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)

    save_image(x_hat.cpu(), os.path.join(subfolder_path, f"reconstruction_{i}.png"))
