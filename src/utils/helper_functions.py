import os
from pathlib import Path
import gdown
import zipfile
import shutil


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
