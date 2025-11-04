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
    batch_size=4,
    micro_image_dataset=True,
)

data_module = data_module.setup()

print(Fore.BLUE + "dataset loaded successfully\n" + Fore.RESET)

print(Fore.BLUE + "loading model..." + Fore.RESET)


model = VQVAE(
    h_dim=56,
    res_h_dim=128,
    n_res_layers=5,
    n_embeddings=2056,
    embedding_dim=1,
    beta=0.25,
    lr=1e-3,
    num_downsamples=3,
    initial_channels=128,
)


reconstruction_callback = ReconstructionCallback(
    input_image_path="images/compress_test.png",
    log_every_n_steps=100,
)

save_callback = ModelCheckpoint(
    monitor="train_loss",
    dirpath="./checkpoints",
    filename="vq_vae_fda_model_final",
    save_top_k=1,
)

logger = WandbLogger(
    project="LIMPACT - VQ-VAE FDA",
    log_model="all",
)

print(Fore.BLUE + "model loaded successfully\n" + Fore.RESET)

print(Fore.BLUE + "starting training..." + Fore.RESET)

trainer = Trainer(
    max_epochs=-1,
    precision="bf16-mixed",
    accelerator="auto",
    devices="auto",
    callbacks=[reconstruction_callback, save_callback],
    logger=logger,
)

trainer.fit(model, datamodule=data_module)
