import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from src.data_preprocessing.dataset import FieldRoadDataset
from sklearn.model_selection import StratifiedKFold
from collections.abc import Callable
import os
from glob import glob


class FieldRoadDatasetKFold:
    def __init__(
        self,
        root_dir: str = "dataset/train",
        transforms_train: Callable | None = None,
        transforms_eval: Callable | None = None,
        batch_size: int = 32,
        image_size: int = 224,
        nb_folds: int = 5,
        split_seed: int = 42,
        weighted_sampler: bool = True,
    ):
        self.root_dir = root_dir
        self.transforms_train = transforms_train
        self.transforms_eval = transforms_eval
        self.batch_size = batch_size
        self.image_size = image_size
        self.nb_folds = nb_folds
        self.split_seed = split_seed

        self.class_names = [class_name for class_name in os.listdir(root_dir)]
        self.class_names_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }
        self.files, self.labels = [], []
        for class_name in self.class_names:
            for f in glob(f"{os.path.join(self.root_dir, class_name)}/*"):
                self.files.append(f)
                self.labels.append(class_name)

        self.weighted_sampler = weighted_sampler

    def compute_weights_for_each_image(self):
        class_counts = [0] * len(self.class_names)
        for label in self.labels:
            class_counts[self.class_names_to_idx[label]] += 1

        self.weights = [0] * len(self.labels)
        for idx, label in enumerate(self.labels):
            self.weights[idx] = 1 / class_counts[self.class_names_to_idx[label]]

    def setup(self) -> None:
        kf = StratifiedKFold(
            n_splits=self.nb_folds,
            shuffle=True,
            random_state=self.split_seed,
        )
        self.folds = [fold for fold in kf.split(self.files, self.labels)]
        self.compute_weights_for_each_image()

    def train_dataloader(self, fold_index: int) -> DataLoader:
        train_indexes = self.folds[fold_index][0]
        train_dataset = FieldRoadDataset(
            files=[self.files[idx] for idx in train_indexes],
            labels=[self.labels[idx] for idx in train_indexes],
            class_names_to_idx=self.class_names_to_idx,
            transforms=self.transforms_train,
        )
        weighted_sampler = WeightedRandomSampler(
            weights=[self.weights[idx] for idx in train_indexes],
            num_samples=len(train_dataset),
            replacement=True,
        )

        return (
            DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=weighted_sampler,
            )
            if self.weighted_sampler
            else DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )
        )

    def val_dataloader(self, fold_index: int) -> DataLoader:
        val_indexes = self.folds[fold_index][1]
        val_dataset = FieldRoadDataset(
            files=[self.files[idx] for idx in val_indexes],
            labels=[self.labels[idx] for idx in val_indexes],
            class_names_to_idx=self.class_names_to_idx,
            transforms=self.transforms_eval,
        )

        return DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


def test_kfold():
    dataset = FieldRoadDatasetKFold()
    dataset.setup()
    for fold_index in range(dataset.nb_folds):
        print(f"Fold {fold_index}")
        nb_train_samples = len(dataset.folds[fold_index][0])
        nb_val_samples = len(dataset.folds[fold_index][1])
        nb_fields_samples_in_train = sum(
            [
                1
                for label in [
                    dataset.labels[idx] for idx in dataset.folds[fold_index][0]
                ]
                if label == "fields"
            ]
        )
        nb_roads_samples_in_train = sum(
            [
                1
                for label in [
                    dataset.labels[idx] for idx in dataset.folds[fold_index][0]
                ]
                if label == "roads"
            ]
        )
        nb_fields_samples_in_val = sum(
            [
                1
                for label in [
                    dataset.labels[idx] for idx in dataset.folds[fold_index][1]
                ]
                if label == "fields"
            ]
        )
        nb_roads_samples_in_val = sum(
            [
                1
                for label in [
                    dataset.labels[idx] for idx in dataset.folds[fold_index][1]
                ]
                if label == "roads"
            ]
        )

        print(
            f"{nb_fields_samples_in_train*100/nb_train_samples:.2f}% fields samples in train\n{nb_roads_samples_in_train*100/nb_train_samples:.2f}% roads samples in train\n"
        )
        print(
            f"{nb_fields_samples_in_val*100/nb_val_samples:.2f}% fields samples in val\n{nb_roads_samples_in_val*100/nb_val_samples:.2f}% roads samples in val\n"
        )


def test_weighted_sampler():
    dataset = FieldRoadDatasetKFold(
        transforms_train=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.Resize((224, 224)),
            ]
        ),
        transforms_eval=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.Resize((224, 224)),
            ]
        ),
    )
    dataset.setup()
    print("With weighted sampler")
    for fold_index in range(dataset.nb_folds):
        print(f"Fold {fold_index}")
        train_loader = dataset.train_dataloader(fold_index)

        for i, (_, labels) in enumerate(train_loader):
            print(f"\tBatch {i}")
            class_counts = [0] * len(dataset.class_names)
            for i in range(len(dataset.class_names)):
                class_counts[i] = torch.sum(labels == i).item()
            for i in range(len(class_counts)):
                print(
                    f"\t\t{dataset.class_names[i]}: {class_counts[i]*100/len(labels):.2f}%"
                )

    # without weighted sampler
    print("\Without weighted sampler")
    dataset = FieldRoadDatasetKFold(
        transforms_train=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.Resize((224, 224)),
            ]
        ),
        transforms_eval=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.Resize((224, 224)),
            ]
        ),
        weighted_sampler=False,
    )
    dataset.setup()
    for fold_index in range(dataset.nb_folds):
        print(f"Fold {fold_index}")
        train_loader = dataset.train_dataloader(fold_index)

        for i, (_, labels) in enumerate(train_loader):
            print(f"\tBatch {i}")
            class_counts = [0] * len(dataset.class_names)
            for i in range(len(dataset.class_names)):
                class_counts[i] = torch.sum(labels == i).item()
            for i in range(len(class_counts)):
                print(
                    f"\t\t{dataset.class_names[i]}: {class_counts[i]*100/len(labels):.2f}%"
                )


if __name__ == "__main__":
    test_weighted_sampler()


# def make_weights_for_balanced_classes(images, nclasses):
#     count = [0] * nclasses
#     # count the number of images in each class
#     for item in images:
#         count[item[1]] += 1

#     weight_per_class = [0.0] * nclasses
#     N = float(sum(count))  # total number of images
#     # calculate the weight for each class: w = total_samples / num_samples_in_class
#     for i in range(nclasses):
#         weight_per_class[i] = N / float(count[i])
#     weight = [0] * len(images)  # weight for each image
#     # assign the weight for each image in the dataset based on the class it belongs to
#     for idx, val in enumerate(images):
#         weight[idx] = weight_per_class[val[1]]
#     return weight


# # And after this, use it in the next way:

# dataset_train = datasets.ImageFolder(traindir)

# # For unbalanced dataset we create a weighted sampler
# weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
# weights = torch.DoubleTensor(weights)
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

# train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle = True,
#                                                              sampler = sampler, num_workers=args.workers, pin_memory=True)
