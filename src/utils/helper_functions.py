import os
from pathlib import Path
import gdown
import zipfile
import shutil
from collections.abc import Callable
from torchvision import transforms


def download_and_unzip_data(
    url: str,
    destination: str,
) -> None:
    """Download and unzip dataset from drive url

    Args:
        url: the url of the dataset
        destination: the destination folder to store the dataset
    """
    # make sure the destination folder exists
    Path(destination).mkdir(parents=True, exist_ok=True)

    # download the zip file
    zip_path = os.path.join(destination, "dataset.zip")
    gdown.download(url, zip_path, quiet=False)

    # extract the zip file
    # the zipfile is extracted to the `dataset` folder
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination)

    # remove the zip file
    os.remove(zip_path)

    data_folder = os.path.join(destination, "dataset")

    # put `fields` and `roads` folders in the `train` folder
    train_folder = os.path.join(data_folder, "train")
    Path(train_folder).mkdir(parents=True, exist_ok=True)
    shutil.move(os.path.join(data_folder, "fields"), train_folder)
    shutil.move(os.path.join(data_folder, "roads"), train_folder)


def get_image_transforms() -> Callable:
    """
    get the transformations to apply to the images
    """
    return transforms.Compose(
        [
            transforms.Resize(
                size=(224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                max_size=None,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_image_augmentations() -> Callable:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=(0.6, 1.4),
                contrast=(0.6, 1.4),
                saturation=(0.6, 1.4),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
