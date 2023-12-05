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

To test the hypothesis that pre-trained model are good at extract meaningful features, we can attempt to extract embedding vectors of all training images, and plot them in a 2D space using PCA. We expect that images of the same class will be clustered together (the visualization may not be perfect due to the loss of information resulting from reducing the embedding to a 2D space).

<img src="plots/test.html" alt="Embedding PCA Plot">



## Training with Support Vector Classification

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

## Future
- [ ] Test CLIP
- [ ] Test SVM without training: Compare CNN features vs CLIP's image encoder features
