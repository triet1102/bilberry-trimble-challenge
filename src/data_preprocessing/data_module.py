from torch.utils.data import DataLoader
from src.data_preprocessing.dataset import FieldRoadDataset
from sklearn.model_selection import StratifiedKFold
from collections.abc import Callable
import os
from glob import glob


class FieldRoadDatasetKFold:
    def __init__(
        self,
        root_dir: str = "dataset/train",
        transforms: Callable | None = None,
        batch_size: int = 32,
        image_size: int = 224,
        nb_folds: int = 5,
        split_seed: int = 42,
    ):
        self.root_dir = root_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.image_size = image_size
        self.nb_folds = nb_folds
        self.split_seed = split_seed

        self.class_names = [class_name for class_name in os.listdir(root_dir)]
        self.files, self.labels = [], []
        for class_name in self.class_names:
            for f in glob(f"{os.path.join(self.root_dir, class_name)}/*"):
                self.files.append(f)
                self.labels.append(class_name)

    def setup(self) -> None:
        kf = StratifiedKFold(
            n_splits=self.nb_folds,
            shuffle=True,
            random_state=self.split_seed,
        )
        self.folds = [fold for fold in kf.split(self.files, self.labels)]

    def train_dataloader(self, fold_index: int) -> DataLoader:
        train_indexes = self.folds[fold_index][0]
        train_dataset = FieldRoadDataset(
            files=[self.files[idx] for idx in train_indexes],
            labels=[self.labels[idx] for idx in train_indexes],
        )

        return DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self, fold_index: int) -> DataLoader:
        val_indexes = self.folds[fold_index][1]
        val_dataset = FieldRoadDataset(
            files=[self.files[idx] for idx in val_indexes],
            labels=[self.labels[idx] for idx in val_indexes],
        )

        return DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
