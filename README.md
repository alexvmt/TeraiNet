# Tiger classification

This repository contains scripts and notebooks to build a model that can classify tigers (and other species) in camera trap images,
using ML (e. g. [MegaDetector](https://github.com/agentmorris/MegaDetector)), open data (e. g. [LILA BC](https://lila.science/)),
open source tools (e. g. [MEWC](https://github.com/zaandahl/mewc)) and free compute resources (i. e. Colab and Kaggle).

![tiger](media/anno_1440.jpg 'tiger')

*Credentials: LILA BC, MegaDetector, own illustration.*


## Motivation and relevance

- tigers are an endangered species,
NGOs like the [Nepal Tiger Trust](https://www.nepaltigertrust.org/) protect them
- there is no open and easy way for ecologists/researchers/NGOs
to classify their camera trap images with regard to tigers
- ML and open data/tools can help reduce the amount of manual labor
when sifting through large amounts of camera trap images, looking for the needle in the haystack
- goal: train a species classifier for Nepal (focussing on tigers)
and make it available through [EcoAssist](https://addaxdatascience.com/ecoassist/)

## Data

**Data sources**

- [LILA BC](https://lila.science/)
- amur tiger re-identification [challenge](https://cvwc2019.github.io/challenge.html) at CVWC 2019

**Sample and download images**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexvmt/tiger_classification/blob/main/notebooks/sample_and_download_images.ipynb)

1. Download image URLs and labels from LILA BC
2. For each selected species: sample and download images, create train test split if applicable
3. Copy images to Drive

*Note: Since Colab and Drive have limited capacities, one might have to further split up the process.*

*Note: I found the image downloading to be much faster in Colab and Drive compared to Kaggle.*

**Preprocess images**

[Open in Kaggle](https://www.kaggle.com/code/alexvmt/preprocess-images/notebook)

1. Run MegaDetector on all images
2. Snip images following [mewc-snip](https://github.com/zaandahl/mewc-snip)
3. Copy snipped images to Kaggle Output

*Note: Images must have been previously downloaded to Drive via Colab and then uploaded to Kaggle (zipped folder).*

*Note: I found access to free GPUs much better and transparent in Kaggle compared to Colab.*

## Training

[Open in Kaggle](https://www.kaggle.com/code/alexvmt/training/notebook)

1. Use [Keras Image Models](https://github.com/james77777778/keras-image-models)
2. Follow [mewc-train](https://github.com/zaandahl/mewc-train)
3. Log experiments using [Weights & Biases](https://wandb.ai/alexvmt/tiger_classification/overview)

I selected a pre-trained EfficientNetV2S with 21 mio parameters because it constitutes a good compromise between predictive performance, training time and model size.
The model has been trained for 30 epochs (early stopping after 24 epochs) with 4000 images per class.
The model has been evaluated on 300 images per class. Below is the resulting confusion matrix.

![confusion_matrix](media/confusion_matrix.png 'confusion_matrix')

Other metrics can be found in the [respective experiment run](https://wandb.ai/alexvmt/tiger_classification/runs/0an7w90t/overview) on Weights & Biases.

*Note: There are only ~300 tiger images on LILA BC. I didn't use them in training but instead put all of them in `test2`
to examine how the model would potentially generalize to tiger camera trap images from another source than the tiger training images
(like it would be the case with the Nepal Tiger Trust using the model on their own images through EcoAssist).*

## Deployment

1. Publish model on [HuggingFace](https://huggingface.co/alexvmt/tiger_classification/tree/main)
2. Integrate and use model in [EcoAssist](https://addaxdatascience.com/ecoassist/)

Join [AI for Conservation Slack](https://beerys.github.io/#slack) and [WILDLABS](https://wildlabs.net/) if you're interested in using technology for conservation.

Feel free to reach out if you have feedback/ideas or would like to contribute/collaborate!
