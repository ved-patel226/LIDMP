import torch

torch.set_float32_matmul_precision("high")

import os
import sys
from art import text2art
from colorama import Fore, init
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

seed_everything(1024)
init()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import ImageCompressionDataModule
from src.models.compression.vq_vae import VQVAE
from callbacks.reconstruction_callback import ReconstructionCallback

print(Fore.GREEN + text2art("Two-Stage Training") + Fore.RESET)
print(Fore.GREEN + "Ved Patel - All Rights Reserved\n" + Fore.RESET)

# Load dataset
print(Fore.BLUE + "Loading dataset..." + Fore.RESET)
data_module = ImageCompressionDataModule(
    data_dir="./data/PetImages",
    batch_size=4,
)
data_module = data_module.setup()
print(Fore.BLUE + "Dataset loaded successfully\n" + Fore.RESET)

# ============================================
# STAGE 1: Pretrain Quantizer Only
# ============================================
print(Fore.YELLOW + "\n=== STAGE 1: Pretraining Quantizer ===" + Fore.RESET)

model = VQVAE(
    h_dim=256,
    n_embeddings=1024,
    embedding_dim=64,
    beta=0.25,
    lr=1e-3,
    n_heads=4,
    n_layers=3,
    patch_size=1,
    freeze_encoder=False,  # Freeze encoder (random initialization)
    freeze_decoder=False,  # Freeze decoder (random initialization)
    freeze_quantizer=False,  # Train quantizer only
    gradient_descent_quantizer=True,
)

checkpoint_callback_stage1 = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints/stage1",
    filename="vq_vae_quantizer_pretrain",
    save_top_k=1,
    mode="min",
)

logger_stage1 = WandbLogger(
    project="LIMPACT - VQ-VAE",
    name="stage1-quantizer-pretrain",
    log_model="all",
)

trainer_stage1 = Trainer(
    max_epochs=-1,
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    callbacks=[checkpoint_callback_stage1],
    logger=logger_stage1,
    val_check_interval=0.5,
    gradient_clip_val=1.0,
)

print(Fore.BLUE + "Starting Stage 1 training..." + Fore.RESET)
trainer_stage1.fit(model, datamodule=data_module)
