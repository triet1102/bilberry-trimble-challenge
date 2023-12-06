import os
from pathlib import Path
import gdown
import zipfile
import shutil
from collections.abc import Callable

import torch
import timm
from torchvision import transforms
import pandas as pd
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from PIL import Image


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


def get_backbone(config: dict) -> torch.nn.Module:
    """get the model and transforms

    Args:
        config: the configuration
        file_path: if provided, load the trained model from the file path
                   if not provided, load the model from timm

    Returns:
        the model and train/val transforms
    """
    backbone = timm.create_model(
        model_name=config.model.backbone,
        pretrained=True,
        num_classes=0,
    )

    if config.model.freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone


def dict_to_yaml(data: dict[str, int], path: str):
    """Non-nested dictionary to yaml file"""
    with open(path, "w") as yaml_file:
        yaml_content = "\n".join([f"{key}: {value}" for key, value in data.items()])
        yaml_file.write(yaml_content)


def find_index_of_config(
    df: pd.DataFrame,
    config: dict,
) -> int | None:
    """Get the index of best model hyperparams in the dataframe."""
    for idx, hyperparams in enumerate(df["params"]):
        parsed_dict = ast.literal_eval(hyperparams)
        if config == parsed_dict:
            return idx

    print("Config not found")
    return


def plot_predictions(
    file_names: list[str],
    predictions: list[str],
    class_names: list[str],
    save_path: str,
) -> go.Figure:
    """Save the predictions of the model on the test dataset"""
    nb_images = len(file_names)
    nb_cols = 3
    nb_rows = math.ceil(nb_images / nb_cols)  # 3 images per row

    titles = [
        f"Image {file_names[i].split('/')[-1]} => predicted {class_names[predictions[i]]}"
        for i in range(nb_images)
    ]
    # titles = [f"Image {i+1}" for i in range(nb_images)]
    figure = make_subplots(
        rows=nb_rows,
        cols=nb_cols,
        subplot_titles=titles,
    )

    for row in range(nb_rows):
        for col in range(nb_cols):
            image_index = row * nb_cols + col
            if image_index >= nb_images:
                pass
            else:
                file = file_names[image_index]
                image = Image.open(file)
                figure.add_trace(
                    go.Image(
                        z=image,
                        xaxis=None,
                        yaxis=None,
                    ),
                    row=row + 1,
                    col=col + 1,
                )

    figure.update_layout(
        height=800,  # Adjust the height of the entire figure
        width=1200,  # Adjust the width of the entire figure
    )
    figure.update_xaxes(showticklabels=False)
    figure.update_yaxes(showticklabels=False)

    figure.write_image(save_path)

    return figure
