import torch
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from pathlib import Path
from src.data_preprocessing.data_module import FieldRoadDatasetKFold


def extract_features_and_plot(
    model: torch.nn.Module,
    data_module: FieldRoadDatasetKFold,
    save_plot: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    # get data
    dataloader = data_module.all_dataloader()

    output_features = []
    output_labels = []

    # extract features
    for data, label in dataloader:
        with torch.no_grad():
            output = model(data)  # dim

        output_features.append(output)
        output_labels.append(label)

    output_features = torch.cat(output_features)
    output_labels = torch.cat(output_labels)

    output_features = output_features.numpy()
    output_labels = output_labels.numpy()

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(output_features)

    if save_plot:
        fig = go.Figure()

        for class_name in data_module.class_names:
            indexes = [
                i
                for i in range(len(data_module.labels))
                if data_module.labels[i] == class_name
            ]
            fig.add_trace(
                go.Scatter(
                    x=[pca_features[idx, 0] for idx in indexes],
                    y=[pca_features[idx, 1] for idx in indexes],
                    mode="markers+text",
                    name=class_name,
                    text=[
                        f"{data_module.files[idx].split('/')[-1]}" for idx in indexes
                    ],
                    textposition="bottom center",
                )
            )

        Path.mkdir(Path("plots"), parents=True, exist_ok=True)
        fig.write_html(
            "plots/train_embeddings.html", full_html=False, include_plotlyjs="cdn"
        )

    return output_features, output_labels
