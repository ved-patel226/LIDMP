import os
import torch
import pytorch_lightning as pl
from PIL import Image
import numpy as np


class ReconstructionCallback(pl.Callback):
    def __init__(self, input_image_path, output_dir, filename="reconstructed.png"):
        super().__init__()
        self.input_image_path = input_image_path
        self.base_output_dir = output_dir
        self.output_dir = self._get_versioned_dir()
        self.filename = filename
        os.makedirs(self.output_dir, exist_ok=True)
        self.image = self._load_image()
        self.epoch_counter = 0

    def _get_versioned_dir(self):
        version = 1
        while True:
            versioned_dir = os.path.join(self.base_output_dir, f"v{version}")
            if not os.path.exists(versioned_dir):
                return versioned_dir
            version += 1

    def _load_image(self):
        img = Image.open(self.input_image_path).convert("RGB")
        img = np.array(img) / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return img

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            output = pl_module(self.image.to(pl_module.device))
            _, output, _ = output
            output = output.squeeze().cpu().permute(1, 2, 0).numpy()
            output = ((output + 1) * 127.5).clip(0, 255).astype(np.uint8)
            out_img = Image.fromarray(output)
            self.epoch_counter += 1
            out_path = os.path.join(self.output_dir, f"{self.epoch_counter}.png")
            out_img.save(out_path)
            print(f"Reconstructed image saved to {out_path}")
