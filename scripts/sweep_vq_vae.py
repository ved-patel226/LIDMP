import torch

torch.set_float32_matmul_precision("high")

import os
import sys
from art import text2art
from colorama import Fore, init
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning import seed_everything
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp import CPUOffload, BackwardPrefetch, ShardingStrategy
import wandb


seed_everything(1024)

init()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.dataset import ImageCompressionDataModule
from src.models.compression.vq_vae import VQVAE
from callbacks.reconstruction_callback import ReconstructionCallback

print(Fore.GREEN + text2art("Training Starting!") + Fore.RESET)
print(Fore.GREEN + "Ved Patel - All Rights Reserved\n" + Fore.RESET)


def train():
    wandb.init()
    config = wandb.config

    print(Fore.BLUE + "loading dataset..." + Fore.RESET)

    data_module = ImageCompressionDataModule(
        data_dir="./data/PetImages",
        batch_size=2,
    )

    data_module = data_module.setup()

    print(Fore.BLUE + "dataset loaded successfully\n" + Fore.RESET)

    print(Fore.BLUE + "loading model..." + Fore.RESET)

    model = VQVAE(
        h_dim=256,
        n_embeddings=config.n_embeddings,
        embedding_dim=config.embedding_dim,
        beta=0.25,
        lr=1e-3,
        n_heads=8,
        n_layers=8,
        patch_size=4,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename=f"vq_vae_nemb{config.n_embeddings}_dim{config.embedding_dim}",
        save_top_k=1,
        mode="min",
    )

    reconstruction_callback = ReconstructionCallback(
        input_image_path="images/compress_test.png",
    )

    logger = WandbLogger(
        project="LIMPACT - VQ-VAE",
        log_model="all",
    )

    print(Fore.BLUE + "model loaded successfully\n" + Fore.RESET)

    print(Fore.BLUE + "starting training..." + Fore.RESET)

    trainer = Trainer(
        max_epochs=2,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=[checkpoint_callback, reconstruction_callback],
        logger=logger,
        val_check_interval=0.5,
        gradient_clip_val=1.0,
    )

    logger.watch(model, log="all", log_freq=25)

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    sweep_config = {
        "method": "grid",
        "metric": {"name": "train_loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 2,
        },
        "parameters": {
            "n_embeddings": {"values": [1024, 2048]},
            "embedding_dim": {"values": [8, 16, 32, 64, 128, 256, 512]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="LIMPACT - VQ-VAE")

    wandb.agent(sweep_id, function=train, count=12)
