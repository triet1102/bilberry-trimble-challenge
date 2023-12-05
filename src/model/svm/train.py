from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import pandas as pd

import timm
from src.data_preprocessing.data_module import FieldRoadDatasetKFold
from src.utils.features import extract_features_and_plot

import matplotlib.pyplot as plt

from glob import glob

from PIL import Image

import torch


# TODO
# 1. Load the data
# 2. Extract features using the pretrained model
# 3. Train the SVM classifier using nested GridSearchCV
# 4. Evaluate the model
def grid_search_svm(
    features: np.ndarray,
    labels: np.ndarray,
):
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
    svm_clf = svm.SVC(class_weight="balanced")
    grid_searcher = GridSearchCV(
        estimator=svm_clf,
        param_grid=param_grid,
        scoring=["accuracy", "precision", "recall", "f1"],
        refit="f1",
        verbose=3,
    )
    grid_searcher.fit(X=features, y=labels)

    cv_results_df = pd.DataFrame(grid_searcher.cv_results_)
    cv_results_df.to_csv("svm_grid_search_results.csv")

    cv_best_params = pd.DataFrame(
        list(grid_searcher.best_params_.items()), columns=["Parameters", "Value"]
    )
    cv_best_params.to_csv("svm_grid_search_best_params.csv")

    return grid_searcher.best_estimator_


def evaluate(
    estimator: svm.SVC,
    features: np.ndarray,
    labels: np.ndarray,
    files: list[str],
    class_names: list[str],
):
    """
    Visualization given an SVC estimator:
        Draw ROC, PR curve with AUC value
        Draw confusion matrix
        Save wrong classified samples
    """
    # ROC curve
    plt.figure()
    metrics.RocCurveDisplay.from_estimator(estimator, features, labels)
    plt.savefig("svm_roc_curve.png")
    plt.close()

    # PR curve
    plt.figure()
    metrics.PrecisionRecallDisplay.from_estimator(estimator, features, labels)
    plt.savefig("svm_pr_curve.png")
    plt.close()

    # Confusion matrix
    plt.figure()
    metrics.ConfusionMatrixDisplay.from_estimator(estimator, features, labels)
    plt.savefig("svm_conf_matrix.png")

    # log all label predictions
    predictions = estimator.predict(features)
    predictions_df = pd.DataFrame(
        {
            "files": files,
            "predictions": [class_names[prediction] for prediction in predictions],
            "labels": [class_names[label] for label in labels],
        }
    )
    predictions_df.to_csv("svm_predictions_on_train_data.csv")

    # log all label predictions on the test dataset
    files_test = [f for f in glob("dataset/test_images/*")]
    features_test = []
    model = timm.create_model(model_name="convnextv2_tiny.fcmae", pretrained=True)
    model.eval()
    # get model specific transforms
    data_config = timm.data.resolve_model_data_config(model)
    transforms_eval = timm.data.create_transform(**data_config, is_training=False)

    for file in files_test:
        image = Image.open(file)
        image = transforms_eval(image)
        with torch.no_grad():
            feature = model(image.unsqueeze(0))
        features_test.append(feature.squeeze(0).numpy())

    features_test = np.array(features_test)
    print(features_test.shape)
    predictions_test = estimator.predict(features_test)
    print(predictions_test.shape)
    predictions_test_df = pd.DataFrame(
        {
            "files": files_test,
            "predictions": [class_names[prediction] for prediction in predictions_test],
        }
    )
    predictions_test_df.to_csv("svm_predictions_on_test_data.csv")


def test(config):
    model = timm.create_model(model_name="convnextv2_tiny.fcmae", pretrained=True)
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms_eval = timm.data.create_transform(**data_config, is_training=False)

    dataset_k_fold = FieldRoadDatasetKFold(transforms_eval=transforms_eval)
    dataset_k_fold.setup()
    features, labels = extract_features_and_plot(
        model=model, data_module=dataset_k_fold, save_plot=False
    )
    print(f"Features shape: {features.shape}\nLabels shape: {labels.shape}")
    best_estimator = grid_search_svm(features, labels)

    evaluate(
        best_estimator,
        features,
        labels,
        dataset_k_fold.files,
        dataset_k_fold.class_names,
    )


if __name__ == "__main__":
    test()
