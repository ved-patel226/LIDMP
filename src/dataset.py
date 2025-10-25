from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms as T
from tqdm import tqdm


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
            raise ValueError(f"Unknown label name: {label_name}")
        return sample, label


class CatDogDataModule(LightningDataModule):
    """
    DataModule for loading Cat and Dog images.
    Image: Resized to (448, 448), RGB converted, Tensor transformed.
    Label: Returned as int, Cat = 0, Dog = 1.
    """

    def __init__(
        self,
        data_dir: str = "./data/PetImages",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_files(self, data_dir, categories, load_content=True):
        default_transform = T.Compose(
            [
                T.Resize((448, 448)),
                T.Lambda(lambda img: img.convert("RGB")),
                T.ToTensor(),
            ]
        )

        dataset = LabelNameImageFolder(
            root=data_dir,
            transform=default_transform,
        )
        if not load_content:
            dataset.samples = [
                (path, label)
                for path, label in dataset.samples
                if dataset.classes[label] in categories
            ]
        return dataset

    def setup(self, stage=None):
        dataset = self.load_files(
            self.data_dir, categories=["cats", "dogs"], load_content=True
        )

        total_size = len(dataset)
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )
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
