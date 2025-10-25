import torch

torch.set_float32_matmul_precision("high")

import os
import sys
from art import text2art
from colorama import Fore, init
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

init()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.dataset import CatDogDataModule
from src.models.cat_dog_model import CatDogModel


print(Fore.GREEN + text2art("Training Starting!") + Fore.RESET)
print(Fore.GREEN + "Ved Patel - All Rights Reserved\n" + Fore.RESET)


print(Fore.BLUE + "loading dataset..." + Fore.RESET)


data_module = CatDogDataModule(
    data_dir="./data/PetImages",
)

data_module = data_module.setup()

print(Fore.BLUE + "dataset loaded successfully\n" + Fore.RESET)

print(Fore.BLUE + "loading model..." + Fore.RESET)

model = CatDogModel()

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="cat_dog_model_best",
    save_top_k=1,
    mode="min",
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    mode="min",
)

logger = WandbLogger(
    project="Impact of Learned Image Compression on Downstream Model Performance",
    log_model="all",
)

print(Fore.BLUE + "model loaded successfully\n" + Fore.RESET)

print(Fore.BLUE + "starting training..." + Fore.RESET)

trainer = Trainer(
    max_epochs=-1,
    accelerator="auto",
    devices="auto",
    callbacks=[checkpoint_callback, early_stopping],
    logger=logger,
)

logger.watch(model, log="all", log_freq=100)

trainer.fit(model, datamodule=data_module)

os.system("shutdown now")
