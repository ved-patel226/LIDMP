import os
import torch
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import wandb


class ReconstructionCallback(pl.Callback):
    def __init__(self, input_image_path, log_every_n_steps=500):
        super().__init__()
        self.input_image_path = input_image_path
        self.image = self._load_image()
        self.input_logged = False
        self.log_every_n_steps = log_every_n_steps

    def _load_image(self):
        img = Image.open(self.input_image_path).convert("RGB")
        img = np.array(img) / 255.0
        img = (
            np.array(
                Image.fromarray((img * 255).astype(np.uint8)).resize(
                    (448, 448), Image.BICUBIC
                )
            )
            / 255.0
        )
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        img = (img - 0.5) / 0.5
        return img

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if global_step % self.log_every_n_steps == 0:
            pl_module.eval()
            with torch.no_grad():
                _, output, _ = pl_module(self.image.to(pl_module.device))
                output = output.squeeze().cpu().permute(1, 2, 0).numpy()
                output = ((output + 1) * 127.5).clip(0, 255).astype(np.uint8)

                log_dict = {
                    "reconstruction/output": wandb.Image(
                        output, caption=f"Step {global_step}"
                    ),
                }

                if not self.input_logged:
                    input_img = self.image.squeeze().cpu().permute(1, 2, 0).numpy()
                    input_img = ((input_img + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    log_dict["reconstruction/input"] = wandb.Image(
                        input_img, caption="Original"
                    )
                    self.input_logged = True

                trainer.logger.experiment.log(log_dict)
            pl_module.train()
