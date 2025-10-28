from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms as T
from tqdm import tqdm
from torch.nn import functional as F
import torch


class LabelNameImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        label_name = self.classes[target]
        label_lower = label_name.lower()
        if "cat" in label_lower:
            label = 0
        elif "dog" in label_lower:
            label = 1
        else:
            raise ValueError(
                f"LabelNameImageFolder found unexpected label: {label_name}"
            )

        # label = F.one_hot(torch.tensor(label), num_classes=2).float()

        return (sample, label)


class CatDogDataModule(LightningDataModule):
    """
    DataModule for loading Cat and Dog images.
    Image: Resized to (448, 448), RGB converted, Tensor transformed.
    Label: Returned as int, Cat = 0, Dog = 1.
    """

    def __init__(
        self,
        data_dir: str = "./data/PetImages",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_files(self, data_dir):
        dataset = LabelNameImageFolder(root=data_dir)
        return dataset

    def setup(self, stage=None):
        train_transform = T.Compose(
            [
                T.Resize((448, 448)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(7),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.Lambda(lambda img: img.convert("RGB")),
                T.ToTensor(),
            ]
        )

        val_transform = T.Compose(
            [
                T.Resize((448, 448)),
                T.Lambda(lambda img: img.convert("RGB")),
                T.ToTensor(),
            ]
        )

        # Create separate dataset instances with different transforms
        train_full_dataset = self.load_files(self.data_dir)
        train_full_dataset.transform = train_transform

        val_full_dataset = self.load_files(self.data_dir)
        val_full_dataset.transform = val_transform

        total_size = len(train_full_dataset)
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size

        # Split each dataset independently
        train_dataset, _ = random_split(train_full_dataset, [train_size, val_size])
        _, val_dataset = random_split(val_full_dataset, [train_size, val_size])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        return self

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class ImageCompressionDataModule(LightningDataModule):
    """
    DataModule for loading images for compression tasks.
    Image: Resized to (448, 448), RGB converted, Tensor transformed.
    """

    def __init__(
        self,
        data_dir: str = "./data/PetImages",
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_files(self, data_dir):
        dataset = datasets.ImageFolder(root=data_dir)
        return dataset

    def setup(self, stage=None):
        train_transform = T.Compose(
            [
                T.Resize((448, 448)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.Lambda(lambda img: img.convert("RGB")),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        val_transform = T.Compose(
            [
                T.Resize((448, 448)),
                T.Lambda(lambda img: img.convert("RGB")),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Create separate datasets with different transforms
        train_full_dataset = self.load_files(self.data_dir)
        train_full_dataset.transform = train_transform

        val_full_dataset = self.load_files(self.data_dir)
        val_full_dataset.transform = val_transform

        total_size = len(train_full_dataset)
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size

        train_dataset, _ = random_split(train_full_dataset, [train_size, val_size])
        _, val_dataset = random_split(val_full_dataset, [train_size, val_size])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
