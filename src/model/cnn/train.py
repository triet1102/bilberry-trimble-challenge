from src.data_preprocessing.data_module import FieldRoadDatasetKFold
from src.model.cnn.model import ClassificationModel, get_backbone

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from glob import glob
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.utils.helper_functions import get_image_transforms
from omegaconf import OmegaConf

import wandb

pl.seed_everything(42)

def kfold_cross_valid(config: dict):
    # load the data
    data = FieldRoadDatasetKFold(config=config)
    data.setup()

    # Kfold cross-validation
    for fold_idx in tqdm(range(3, config.data.nb_folds)):
        logger = WandbLogger(
            project="image_classification",
            name=f"normal_1_layer_fold_{fold_idx}"
        )
        train_loader = data.train_dataloader(fold_index=fold_idx)
        val_loader = data.val_dataloader(fold_index=fold_idx)

        # TODO: lr monitor callback, early stopping callback
        lr_monitor = LearningRateMonitor(
            logging_interval="epoch"
        )
        # early_stopping_callback = EarlyStopping(
        #     monitor="val_loss",
        #     mode="min",
        #     patience=10,
        #     verbose=True
        # )
        # callbacks = [lr_monitor, early_stopping_callback]
        callbacks = [lr_monitor]
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.training.nb_epochs,
            log_every_n_steps=3,
            callbacks=callbacks
        )
        model = ClassificationModel(config=config)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        wandb.finish()
    

    # plot average test acc
def train_with_all_data(config):
    # use datamodule.all_dataloader_train
    pass

def inference(config: dict, test_data_dir: str = "data/test_images"):
    model = get_backbone(
        config=config,
        file_path=f"src/model/cnn/{config.model.name}-trained.pth",
    )
    model.eval()
    for file_name in glob(f"{test_data_dir}/*"):
        image = Image.open(file_name)
        image_transforms = get_image_transforms()
        image = image_transforms(image)
        with torch.no_grad():
            output = model(image.unsqueeze(0)).squeeze()  ### check squeeze

        pred = F.softmax(output, dim=-1)
        pred_class = torch.argmax(pred, dim=-1)
        print(f"{file_name}:\tpred_class={pred_class}\tpred={pred}")


if __name__ == "__main__":
    config = OmegaConf.load("src/model/cnn/config.yaml")
    kfold_cross_valid(config=config)
    
