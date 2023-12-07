from src.data_preprocessing.data_module import FieldRoadDatasetKFold
from src.model.cnn.model import ClassificationModel

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from glob import glob
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.utils.helper_functions import get_image_transforms, plot_predictions
from omegaconf import OmegaConf
from pathlib import Path
import os
import wandb

pl.seed_everything(42)


def kfold_cross_valid(config: dict):
    """Train the model using kfold cross-validation to search for the best combination of hyperparameters"""

    # load the data
    data = FieldRoadDatasetKFold(config=config)
    data.setup()

    # Kfold cross-validation
    for fold_idx in tqdm(range(config.data.nb_folds)):
        logger = WandbLogger(
            project="image_classification",
            name=f"with_weighted_sampler_and_augmentation_1_layer_fold_{fold_idx}",
        )
        train_loader = data.train_dataloader(fold_index=fold_idx)
        val_loader = data.val_dataloader(fold_index=fold_idx)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks = [lr_monitor]

        trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.training.nb_epochs,
            log_every_n_steps=3,
            callbacks=callbacks,
        )
        model = ClassificationModel(config=config)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        wandb.finish()


def train_with_all_data(config) -> str:
    """Train model on all data with the best combination of hyperparameters searched by kfold cross-validation"""
    data = FieldRoadDatasetKFold(config)
    data.setup()
    dataloader = data.all_dataloader_train()

    logger = WandbLogger(project="image_classification", name="train_with_all_data")

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="src/model/cnn/checkpoints",
        filename="{epoch}-{train_loss:.2f}-{train_BinaryAccuracy:.2f}-{train_BinaryF1Score:.2f}",
        monitor="train_loss",
        mode="min",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # early_stopping = EarlyStopping(
    #     monitor="train_BinaryF1Score",
    #     mode="max",
    #     patience=30,
    #     verbose=True,
    # )

    # callbacks = [checkpoint_callback, lr_monitor, early_stopping]
    callbacks = [checkpoint_callback, lr_monitor]

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.training.nb_epochs,
        log_every_n_steps=3,
        callbacks=callbacks,
    )

    model = ClassificationModel(config)
    print(
        f"Number of parameters of model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    trainer.fit(model, train_dataloaders=dataloader)

    wandb.finish()

    # return the path of the best model for inference
    return checkpoint_callback.best_model_path


def evaluate(
    config: dict,
    model_ckpt_path: str,
):
    """Evaluate the model on the test data

    Args:
        model_ckpt_path: checkpoint path of the trained model
    """
    # define result folder
    result_folder = Path("src/model/cnn/results")
    result_folder.mkdir(parents=True, exist_ok=True)

    # get the model checkpoint
    model = ClassificationModel.load_from_checkpoint(model_ckpt_path)

    # get the data for retrieving the class names
    data = FieldRoadDatasetKFold(config=config)

    # do the inference on test images
    predictions_test = []
    files_test = [f for f in glob("dataset/test_images/*")]
    model.eval()

    test_embeddings = []

    for file_name in files_test:
        image = Image.open(file_name)
        image_transforms = get_image_transforms()
        image = image_transforms(image)
        with torch.no_grad():
            emd = model.backbone(image.unsqueeze(0))
            print(f"emd.shape={emd.shape}")
            test_embeddings.append(emd)
            output = model(image.unsqueeze(0)).squeeze()
            output = output.squeeze()  ### check squeeze
        pred = F.softmax(output, dim=-1)
        pred_class = torch.argmax(pred, dim=-1)
        print(f"{file_name}:\tpred_class={pred_class}\tpred={pred}")

        predictions_test.append(pred_class.item())

    # save the predictions
    plot_predictions(
        file_names=files_test,
        predictions=predictions_test,
        class_names=data.class_names,
        save_path=os.path.join(result_folder, "predictions_on_test_data.png"),
    )


def main(config: dict):
    """Train and evaluate the model"""

    # # kfold cross validation
    # kfold_cross_valid(config=config)

    # # train the model with the best combination of hyperparameters
    # ckpt_path = train_with_all_data(config=config)

    # # evaluate the trained model
    # evaluate(
    #     config=config,
    #     model_ckpt_path=ckpt_path,
    # )
    evaluate(
        config=config,
        model_ckpt_path="src/model/cnn/checkpoints/epoch=15-train_loss=0.10-train_BinaryAccuracy=0.99-train_BinaryF1Score=0.99.ckpt",
    )


if __name__ == "__main__":
    config = OmegaConf.load("src/model/cnn/config.yaml")
    main(config)
