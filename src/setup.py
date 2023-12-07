import timm
import torch
from src.data_preprocessing.data_module import FieldRoadDatasetKFold

from src.utils.features import extract_features_and_plot
from omegaconf import OmegaConf
from src.utils.helper_functions import download_and_unzip_data
import os

torch.manual_seed(42)


def main():
    # download the dataset
    if not os.path.exists("dataset"):
        download_and_unzip_data(
            url="https://drive.google.com/uc?id=1pOKhKzIs6-oXv3SlKrzs0ItHI34adJsT",
            destination=".",
        )
    else:
        print("Dataset already downloaded, doing nothing...")

    # extract and plot features
    config = OmegaConf.load("src/model/cnn/config.yaml")
    model = timm.create_model(
        model_name=config.model.backbone,
        pretrained=True,
    )
    data = FieldRoadDatasetKFold(config=config)

    extract_features_and_plot(
        model=model,
        data_module=data,
        save_plot=True,
    )


if __name__ == "__main__":
    main()
