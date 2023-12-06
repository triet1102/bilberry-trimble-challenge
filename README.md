# Introduction
This repository contains the solution for technical exercise for AI Engineer position at Trimble / Bilberry.
Your goal is to create a two class classifier : **Field** & **Road** using the available data [here](https://drive.google.com/file/d/1pOKhKzIs6-oXv3SlKrzs0ItHI34adJsT/view?usp=sharing).


Instrunctions to run the code are given in [INSTALL.md](INSTALL.md) file.


# Directory structure
```
├── dataset/                        # train and test images
│   ├── test_images/
│   └── train/
├── plots/                          # plots for experiments
├── requirements/                   # dependencies to install
├── src/                            # source files
│   ├── data_preprocessing/         # dataset / datamodule creation
│   ├── model/                      # model creation, training, evaluation
│   ├── utils/                      # helper functions
│   ├── __init__.py
│   ├── main.py
├── INSTALL.md
├── README.md
```

# Solution
## Analysis

Overall, the training dataset has `153` images of 2 classes:
* `Fields`: 45 images, which corresponds to `29.4%` of the dataset
* `Roads`: 108 images, which corresponds to `70.6%` of the dataset

<div style="display: flex;">
  <img src="dataset/train/fields/1.jpg" alt="Image field" width="400"/>
  <img src="dataset/train/roads/1.jpg" alt="Image road" width="400"/>
</div>
<br>

Insights:
<ol>
  <li>The training dataset is quite small, then when training a deep neural network, it should be better to freeze the weights of the model and only train the head of the model to avoid overfitting.</li>
  <li>There is no validation dataset, then we can use cross-validation to evaluate the model.</li>
  <li>The dataset is imbalanced; therefore, during training, images pertaining to the Field class should be sampled more frequently to achieve a balance between the two classes.</li>
</ol>

For 2D image classification problem, general approach is to use a pre-trained CNN model as backbone, add a classifier head on top of it and fine-tune the model on the target dataset. This is useful because the pre-trained model can extract meaningful features from images as it has been trained on a large dataset. However, as we have a small training dataset, it is better to freeze the weights of the backbone and only train the classifier head.

To test the hypothesis that pre-trained model are good at extract meaningful features, we can attempt to extract embedding vectors of all training images, and plot them in a 2D space using PCA. We expect that images of the same class will be clustered together (the visualization may not be perfect due to the loss of information resulting from reducing the embedding to a 2D space). The backbone used for experiments is [ConvNeXt](https://huggingface.co/timm/convnextv2_tiny.fcmae), a state-of-the-art CNN model on ImageNet dataset.
The plots are show below (for interactive plots, please open [this file](plots/train_embeddings.html)):

<div style="display: flex;">
  <img src="plots/train_embeddings.png" alt="training embeddings" width="400"/>
  <img src="plots/train_embeddings_highlight.png" alt="training embeddings with highlight" width="400"/>
</div>

The plots on 2D space are quite good. However, there are some datapoints of `Field` that have strong signal for `Road` class. By checking several images, we can see that there are some images of `Field` that have a `Road` in the background, e.g [image 3](plots/3.jpg) and [image 5](plots/5.jpg) that are circled by black contour. Therefore, we can remove these images from the training dataset. These 2 images are manually removed from the training dataset.


## Dataset splitting
<ol>
<li>As the size of the training dataset is small, then it's hard to properly split the dataset into train/validation/test sets. Therefore, we can use cross-validation to evaluate the model. The dataset is splitted with StratifiedKFold to preserve the class distribution in each fold.</li>
<li>As the training dataset is imbalance, during training, samples of minor class will be sampled more frequent.</li>
</ol>

## Training with Support Vector Classification
The first approach is to use Support Vector Classification (SVC) with features extracted from [ConvNeXt](https://huggingface.co/timm/convnextv2_tiny.fcmae). The model is trained with 5-fold cross-validation, with different settings by GridSearchCV. The best model is selected based on the average f1-score of 5 folds. The best model is then trained on the whole training dataset and evaluated on the test dataset.

The configuration are shown below:

```python
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
```

The configuration and results for the best model are shown below:
```yaml
C: 0.1
degree: 3
gamma: 0.01
kernel: poly
```
<img src="src/model/svm/results/best_results.png" alt="Image field"/>

The predictions on test dataset are shown below:
<img src="src/model/svm/results/predictions_on_test_data.png" alt="SVM test predictionss"/>



## Training with pre-trained ConvNeXt

## TODO
- [ ] Remove image 3 and 5 of field from training
- [ ] Test overfit_batches
- [ ] Find best combination of CNN for K-Fold=5 as same as SVC
- [ ] Retrain with best combination of CNN for K-Fold
- [ ] Write code to generate predictions and plots for SVC and CNN
- [ ] Clean code SVM, write inference function
- [ ] Make sure dataset sampler is good
- [ ] Continue write report
- [ ] Save model checkpoint

## Future work
- [ ] Acquire more data
- [ ] Test Vision-Language model e.g CLIP for zero-shot learning
- [ ] Compare CNN features vs CLIP's image encoder features
