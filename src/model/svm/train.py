from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

import torch
import os

from glob import glob
from PIL import Image

from src.data_preprocessing.data_module import FieldRoadDatasetKFold
from src.utils.features import extract_features_and_plot
from src.utils.helper_functions import (
    get_backbone,
    dict_to_yaml,
    get_image_transforms,
    plot_predictions,
)

from omegaconf import OmegaConf
from joblib import dump, load
from pathlib import Path


def grid_search_svm(
    features: np.ndarray,
    labels: np.ndarray,
):
    """grid search SVM to find the best hyperparameters

    Args:
        features: the extracted features from the backbone
        labels: the labels of the training data
    """
    # define result folder
    result_folder = Path("src/model/svm/results")
    result_folder.mkdir(parents=True, exist_ok=True)

    # define the hyperparameters to search
    param_grid = [
        {"C": [0.1, 1, 10, 100, 1000], "kernel": ["linear"]},
        {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [0.001, 0.01, 0.1, 1],
            "kernel": ["rbf", "sigmoid"],
        },
        {
            "C": [0.1, 1, 10, 100, 1000],
            "degree": [2, 3, 5, 7, 10],
            "gamma": [0.001, 0.01, 0.1, 1],
            "kernel": ["poly"],
        },
    ]

    # init the classifier
    svm_clf = svm.SVC(class_weight="balanced")
    # init the grid search
    grid_searcher = GridSearchCV(
        estimator=svm_clf,
        param_grid=param_grid,
        scoring=["accuracy", "precision", "recall", "f1"],
        refit="f1",
    )
    grid_searcher.fit(X=features, y=labels)

    # Save grid search results
    cv_results_df = pd.DataFrame(grid_searcher.cv_results_)
    cv_results_df.to_csv(os.path.join(result_folder, "grid_search_results.csv"))

    # Write best params found in a yaml file
    cv_best_params = grid_searcher.best_params_
    dict_to_yaml(cv_best_params, os.path.join(result_folder, "best_params.yaml"))

    # save best SVM
    best_estimator = grid_searcher.best_estimator_
    dump(best_estimator, os.path.join(result_folder, "best_SVM.joblib"))


def evaluate(
    estimator: svm.SVC,
    model: torch.nn.Module,
    data_module: FieldRoadDatasetKFold,
) -> None:
    """
    Visualization given an SVC estimator:
        Draw ROC, PR curve with AUC value
        Draw confusion matrix
        Save wrong classified samples
    """
    # define result folder
    result_folder = Path("src/model/svm/results")
    result_folder.mkdir(parents=True, exist_ok=True)

    # log all label predictions on the test dataset
    files_test = [f for f in glob("dataset/test_images/*")]
    features_test = []
    transforms = get_image_transforms()

    model.eval()
    for file in files_test:
        image = Image.open(file)
        image = transforms(image)
        with torch.no_grad():
            feature = model(image.unsqueeze(0))
        features_test.append(feature.squeeze(0).numpy())

    features_test = np.array(features_test)
    predictions_test = estimator.predict(features_test)
    predictions_test_df = pd.DataFrame(
        {
            "files": files_test,
            "predictions": [
                data_module.class_names[prediction] for prediction in predictions_test
            ],
        }
    )
    predictions_test_df.to_csv(
        os.path.join(result_folder, "predictions_on_test_data.csv")
    )

    plot_predictions(
        file_names=files_test,
        predictions=predictions_test,
        class_names=data_module.class_names,
        save_path=os.path.join(result_folder, "predictions_on_test_data.png"),
    )


def main(config):
    model = get_backbone(config)
    data = FieldRoadDatasetKFold(config)
    features, labels = extract_features_and_plot(
        model=model, data_module=data, save_plot=False
    )
    print(f"Features shape: {features.shape}\nLabels shape: {labels.shape}")

    # define result folder
    result_folder = Path("src/model/svm/results")
    result_folder.mkdir(parents=True, exist_ok=True)

    # train the model
    if config.train:
        grid_search_svm(features, labels)

    try:
        best_estimator = load(os.path.join(result_folder, "best_SVM.joblib"))
    except FileNotFoundError:
        print("No saved estimator found. Need to train first!")
        return

    # do the inference
    evaluate(estimator=best_estimator, model=model, data_module=data)


if __name__ == "__main__":
    config = OmegaConf.load("src/model/svm/config.yaml")
    main(config)
