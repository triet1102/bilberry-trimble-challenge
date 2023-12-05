from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from src.data_preprocessing.dataset import FieldRoadDataset
from sklearn.model_selection import StratifiedKFold
import os
from glob import glob


class FieldRoadDatasetKFold:
    def __init__(
        self,
        config: dict,
    ):
        """the dataset for k-fold cross validation

        Args:
            config: the configuration dict
        """
        self.config = config
        self.class_names = ["roads", "fields"]
        self.class_names_to_idx = {"roads": 0, "fields": 1}
        self.files, self.labels = [], []
        for class_name in self.class_names:
            for f in glob(f"{os.path.join(self.config.data.root_dir, class_name)}/*"):
                self.files.append(f)
                self.labels.append(class_name)

    def compute_weights_for_each_image(self):
        class_counts = [0] * len(self.class_names)
        for label in self.labels:
            class_counts[self.class_names_to_idx[label]] += 1

        self.weights = [0] * len(self.labels)
        for idx, label in enumerate(self.labels):
            self.weights[idx] = 1 / class_counts[self.class_names_to_idx[label]]

    def setup(self) -> None:
        kf = StratifiedKFold(
            n_splits=self.config.data.nb_folds,
            shuffle=True,
            random_state=self.config.data.split_seed,
        )
        self.folds = [fold for fold in kf.split(self.files, self.labels)]
        self.compute_weights_for_each_image()

    def train_dataloader(self, fold_index: int) -> DataLoader:
        train_indexes = self.folds[fold_index][0]
        train_dataset = FieldRoadDataset(
            files=[self.files[idx] for idx in train_indexes],
            labels=[self.labels[idx] for idx in train_indexes],
            class_names_to_idx=self.class_names_to_idx,
            augmentation=self.config.data.augmentation,
        )
        weighted_sampler = WeightedRandomSampler(
            weights=[self.weights[idx] for idx in train_indexes],
            num_samples=len(train_dataset),
            replacement=True,
        )

        return (
            DataLoader(
                dataset=train_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                sampler=weighted_sampler,
                drop_last=True,
                num_workers=4,
                persistent_workers=True,
            )
            if self.config.data.weighted_sampler
            else DataLoader(
                dataset=train_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=4,
                persistent_workers=True,
            )
        )

    def val_dataloader(self, fold_index: int) -> DataLoader:
        val_indexes = self.folds[fold_index][1]
        val_dataset = FieldRoadDataset(
            files=[self.files[idx] for idx in val_indexes],
            labels=[self.labels[idx] for idx in val_indexes],
            class_names_to_idx=self.class_names_to_idx,
            augmentation=False,
        )

        return DataLoader(
            dataset=val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )

    def all_dataloader(self) -> DataLoader:
        """Returns a dataloader with all the data
        Used for extracting features from the pretrained model
        """
        dataset = FieldRoadDataset(
            files=self.files,
            labels=self.labels,
            class_names_to_idx=self.class_names_to_idx,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
