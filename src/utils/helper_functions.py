import os
from pathlib import Path
import gdown
import zipfile
import shutil
from torch import Tensor
import timm
import torch
from torch.utils.data import DataLoader
from glob import glob
from src.data_preprocessing.dataset import FieldRoadDataset
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from pathlib import Path


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


def extract_features_and_plot(
    model_ckpt: str = "convnextv2_tiny.fcmae",
    root_dir: str = "dataset/train",
) -> Tensor:
    # get model
    model = timm.create_model(
        model_name=model_ckpt,
        pretrained=True,
    )
    model.eval()

    # get data congiguration and transforms
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(
        **data_config,
        is_training=False,
    )

    # get image image_paths and labels
    class_names = [class_name for class_name in os.listdir(root_dir)]

    files, labels = [], []
    for class_name in class_names:
        for f in glob(f"{os.path.join(root_dir, class_name)}/*"):
            files.append(f)
            labels.append(class_name)

    # create dataset
    dataset = FieldRoadDataset(
        files=files,
        labels=labels,
        class_names_to_idx={
            class_name: idx for idx, class_name in enumerate(class_names)
        },
        transforms=transforms,
    )

    # create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
    )

    output_features = []

    # extract features
    for data, _ in dataloader:
        with torch.no_grad():
            output = model(data)

        output_features.append(output)

    output_features = torch.cat(output_features)

    output_features = output_features.cpu().numpy()

    pca = PCA(n_components=2)
    features_transformed = pca.fit_transform(output_features)

    fig = go.Figure()

    for class_name in class_names:
        indexes = [i for i in range(len(labels)) if labels[i] == class_name]
        fig.add_trace(
            go.Scatter(
                x=[features_transformed[idx, 0] for idx in indexes],
                y=[features_transformed[idx, 1] for idx in indexes],
                mode="markers",
                name=class_name,
                text=[f"{files[idx].split('/')[-1]}" for idx in indexes],
            )
        )

    Path.mkdir(Path("plots"), parents=True, exist_ok=True)
    fig.write_html("plots/test.html")


if __name__ == "__main__":
    extract_features_and_plot()
