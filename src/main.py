import timm
import torch
from src.data_preprocessing.data_module import FieldRoadDatasetKFold

from src.utils.features import extract_features_and_plot
from omegaconf import OmegaConf

torch.manual_seed(42)


def main():
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
