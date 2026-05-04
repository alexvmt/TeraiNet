# TeraiNet

This repository contains scripts and notebooks to build a species classification model focussed on the Terai region in Nepal, namely TeraiNet.
It can classify 10 different classes (including tigers) in camera trap images, using ML ([MegaDetector](https://github.com/agentmorris/MegaDetector) and EfficientNetV2M),
open data ([LILA BC](https://lila.science/)), open source tools ([MEWC](https://github.com/zaandahl/mewc)) and free compute resources (Google Colab and Kaggle).

TeraiNet is supposed to be an open source blueprint to enable others to easily and flexibly build their own species classifiers for their own specific use cases.
All you need is basically a Google and a Kaggle account, and an Internet connection.
In a nutshell, you select the species you're interested in and run the rest of the pipeline with some (hopefully) minor tweaks here and there.

Feel free to reach out if you have feedback/ideas for new use cases or would like to contribute/collaborate. Join [AI for Conservation Slack](https://beerys.github.io/#slack) and [WILDLABS](https://wildlabs.net/) if you're interested in using technology for conservation.

![tiger](media/anno_1440.jpg 'tiger')

*Credentials: LILA BC, MegaDetector, own illustration.*

## Motivation and relevance

- tigers are an endangered species, NGOs like the [Nepal Tiger Trust](https://www.nepaltigertrust.org/) and [WWF Nepal](https://www.wwfnepal.org) protect them
- there is no open and easy way for ecologists/NGOs to classify their camera trap images with regard to tigers
- ML and open data/tools can help reduce the amount of manual labor when sifting through large amounts of camera trap images, looking for the needle in the haystack
- there appears to be no such species classification model focussed on the Terai region in Nepal yet
- **goal**: train a species classifier focussing on the most relevant species in the Bengal tiger's ecosystem in Nepal, namely the Terai region,
and make it openly available through [AddaxAI](https://addaxdatascience.com/addaxai/) (formerly known as EcoAssist)

## Data preparation

To start, either open the notebooks (step 1-3) in order from here or first clone the repo in your Google Drive.
For step 4, just open the notebook in Kaggle directly by clicking the respective button below.

### 1. Select data sources

- [LILA BC](https://lila.science/)
- amur tiger re-identification [challenge](https://cvwc2019.github.io/challenge.html) at CVWC 2019

### 2. Sample images

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexvmt/TeraiNet/blob/main/notebooks/sample_images_from_lila_bc.ipynb)

- Get image URLs from LILA BC
- Sample desired amount of images per class
- Create train test split if applicable

**Classes**

1. Tiger
2. Leopard
3. Asian black bear, American black bear *(not enough images of Asian black bear alone)*
4. Other carnivores, including dhole, black-backed jackal, gray fox, leopard cat, mainland leopard cat, marbled cat, Asian golden cat *(including substitutes, i.e., black-backed jackal and gray fox)*
5. Deer
6. Wild boar
7. African buffalo, Cape buffalo *(substitute for gaur)*
8. White rhinoceros *(substitute for Indian rhino)*
9. Asian elephant, African bush elephant *(not enough images of Asian elephant alone)*
10. Bird

*Note: Due to a lack of images for certain species some fairly heavy compromises had to be taken. It remains to be seen how well a model trained with such compromises generalizes to the target region.*

### 3. Download images

- LILA BC [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexvmt/TeraiNet/blob/main/notebooks/download_images_from_lila_bc.ipynb)
- amur tiger re-identification challenge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexvmt/TeraiNet/blob/main/notebooks/download_images_from_amur_tiger_re_identification_challenge.ipynb)

*Note: Since free Colab and Drive have limited storage capacities, one might have to download images for one species at a time (and free up space before proceeding to the next).*

*Note: I found the image downloading to be much faster in Colab and Drive compared to Kaggle.*

### 4. Preprocess images

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/alexvmt/preprocess-images-for-terainet)

1. Run MegaDetector on all images
2. Snip images following [mewc-snip](https://github.com/zaandahl/mewc-snip)
3. Copy snipped images to Kaggle Output

*Note: Images must have been previously downloaded to Drive/local machine via Colab and then uploaded to Kaggle as a dataset (zipped folder).*

*Note: I found access to free GPUs much better and transparent in Kaggle compared to Colab.*

## Model training and evaluation

To train and evaluate a model, open the notebook in Kaggle directly by clicking the respective button below.

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/alexvmt/train-and-evaluate-terainet/notebook?scriptVersionId=232861979)

1. Use [Keras Image Models](https://github.com/james77777778/keras-image-models)
2. Follow [mewc-flow](https://github.com/zaandahl/mewc-flow/blob/main/requirements.txt) and [mewc-train](https://github.com/zaandahl/mewc-train)
3. Log experiments using [Weights & Biases](https://wandb.ai/alexvmt/TeraiNet/overview)

I selected a pre-trained EfficientNetV2M with 54M parameters because it constitutes a good compromise between predictive performance, training time and model size.
The model has been trained for 50 epochs (early stopping after 43 epochs) with 2,000 images per class (250 images per class in the validation set).
The model has been evaluated on a test set with 250 images per class.

### Test set performance overview

- Accuracy: 0.899
- Precision: 0.899
- Recall: 0.899
- F1: 0.899

### Test set performance per class

| Class            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| tiger            | 1.000     | 1.000  | 1.000    | 250     |
| leopard          | 0.883     | 0.848  | 0.865    | 250     |
| black_bear       | 0.967     | 0.948  | 0.958    | 250     |
| other_carnivores | 0.912     | 0.916  | 0.914    | 250     |
| deer             | 0.893     | 0.868  | 0.880    | 250     |
| wild_boar        | 0.922     | 0.952  | 0.937    | 250     |
| buffalo          | 0.879     | 0.932  | 0.905    | 250     |
| rhino            | 0.863     | 0.832  | 0.847    | 250     |
| elephant         | 0.853     | 0.812  | 0.832    | 250     |
| bird             | 0.818     | 0.880  | 0.848    | 250     |
| **micro avg**    | 0.899     | 0.899  | 0.899    | 2500    |
| **macro avg**    | 0.899     | 0.899  | 0.899    | 2500    |
| **weighted avg** | 0.899     | 0.899  | 0.899    | 2500    |

Test set performance for the tiger class is extremely high because the respective images are unrealistically good.
The performance on the tiger images in `test2` is more realistic (`test2` accuracy: 0.872).
Most other classes have metrics of about 0.87-0.96. But performance on some classes, including rhino, elephant and bird is somewhat lower.
I believe that the quality of the rhino and elephant images is not ideal, e. g. there seem to be mix-ups or even other species included (e. g. zebra).
Birds tend to be difficult to classify correctly in general.

### Test set confusion matrix

![confusion_matrix](media/confusion_matrix.png 'confusion_matrix')

There appears to be some confusion between leopard and rhino, other carnivores and deer, rhino and buffalo, rhino and bird, and elephant and bird.
I believe that there are several instances in the present data where there are birds sitting on elephants, which obviously leads to erronous annotations.

*Note: There are only ~300 tiger images on LILA BC. I didn't use them in training and validation but instead put all of them in `test2`
to examine how the model would potentially generalize to tiger camera trap images from another source than the tiger training images
(like it would be the case with the Nepal Tiger Trust using the model on their own images through AddaxAI, for example).*

## Deployment

1. Publish model on [HuggingFace](https://huggingface.co/alexvmt/TeraiNet/tree/main)
2. Deploy model to [AddaxAI](https://addaxdatascience.com/addaxai/)

There's also an inference example [here](https://www.kaggle.com/code/alexvmt/terainet-inference-example) to use the model directly.

## Cite

[Paper](https://github.com/alexvmt/TeraiNet/blob/main/terainet_paper.pdf)

```BibTeX
@article{Merdian-Tarko2025,
  title = {TeraiNet: An open-source blueprint for the fast and flexible development of local species classification models},
  author = {Alexander V. Merdian-Tarko},
  workshop = {5th International Workshop on Camera Traps, AI, and Ecology},
  location = {Seatle, WA, U.S. & remote},
  year = {2025}
}
```
