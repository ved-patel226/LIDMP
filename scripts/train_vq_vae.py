import torch

torch.set_float32_matmul_precision("high")

import os
import sys
from art import text2art
from colorama import Fore, init
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

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
    batch_size=1,
)

data_module = data_module.setup()

print(Fore.BLUE + "dataset loaded successfully\n" + Fore.RESET)

print(Fore.BLUE + "loading model..." + Fore.RESET)


model = VQVAE(
    h_dim=56,
    res_h_dim=256,
    n_res_layers=3,
    n_embeddings=512 * 2,
    embedding_dim=16,
    beta=0.25,
    lr=1e-3,
    num_downsamples=3,
    initial_channels=128,
)

# model = VQVAE.load_from_checkpoint("./checkpoints/vq_vae_model_best-v13.ckpt")

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="vq_vae_model_best",
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

# DeepSpeed configuration with CPU offloading for maximum memory efficiency
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,  # ZeRO Stage 3 - offload parameters, gradients, and optimizer states
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": False,
}

trainer = Trainer(
    max_epochs=-1,
    accelerator="gpu",
    devices=1,
    strategy=DeepSpeedStrategy(config=deepspeed_config),
    precision="bf16-mixed",
    callbacks=[checkpoint_callback, reconstruction_callback],
    logger=logger,
    val_check_interval=0.5,
    enable_model_summary=False,  # Disable to avoid DeepSpeed compatibility issues
    accumulate_grad_batches=2,  # Accumulate gradients for memory efficiency
)


logger.watch(model, log="all", log_freq=25)

trainer.fit(model, datamodule=data_module)
