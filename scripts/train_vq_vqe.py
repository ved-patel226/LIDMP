import torch

torch.set_float32_matmul_precision("high")

import os
import sys
from art import text2art
from colorama import Fore, init
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning import seed_everything

seed_everything(1024)

init()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.dataset import ImageCompressionDataModule
from src.models.compression.vq_vae import VQVAE
from callbacks.reconstruction_callback import ReconstructionCallback

print(Fore.GREEN + text2art("Training Starting!") + Fore.RESET)
print(Fore.GREEN + "Ved Patel - All Rights Reserved\n" + Fore.RESET)


print(Fore.BLUE + "loading dataset..." + Fore.RESET)


data_module = ImageCompressionDataModule(
    data_dir="./data/PetImages",
)

data_module = data_module.setup()

print(Fore.BLUE + "dataset loaded successfully\n" + Fore.RESET)

print(Fore.BLUE + "loading model..." + Fore.RESET)

model = VQVAE(
    h_dim=128,
    res_h_dim=128,
    n_res_layers=3,
    n_embeddings=512,
    embedding_dim=64,
    beta=0.25,
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="vq_vae_model_best",
    save_top_k=1,
    mode="min",
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    mode="min",
)

reconstruction_callback = ReconstructionCallback(
    input_image_path="images/compress_test.png",
    output_dir="images/reconstructed",
)

logger = WandbLogger(
    project="LIDMP - VQ-VAE",
    log_model="all",
)

print(Fore.BLUE + "model loaded successfully\n" + Fore.RESET)

print(Fore.BLUE + "starting training..." + Fore.RESET)

trainer = Trainer(
    max_epochs=-1,
    accelerator="auto",
    devices="auto",
    callbacks=[checkpoint_callback, early_stopping, reconstruction_callback],
    logger=logger,
)

logger.watch(model, log="all", log_freq=25)

trainer.fit(model, datamodule=data_module)
