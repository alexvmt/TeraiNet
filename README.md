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

Training settings live in [training.yaml](training.yaml). Unlike the shared [config.yaml](config.yaml),
this file is self-contained: it defines the ordered semantic class mapping, the corresponding image
directory names, dataset and artifact paths, sampling/filtering policy, model hyperparameters, and
optional W&B logging. Keep the numeric class indices stable: they are the label contract used for
training, evaluation, saved class lists, and inference.

The training notebook is an orchestrator. It loads and validates the training configuration,
prepares deterministic samples, trains the model, saves the run contract, and evaluates the
held-out test set. It keeps only Kaggle-specific W&B authentication and interactive displays.
Authenticate W&B in the notebook through the Kaggle secret named `wandb` when logging is enabled.

`test` is the held-out in-distribution evaluation set used for the main report. The optional `ood`
directory is evaluated and logged separately; it must not be combined with the test result. When
location metadata exists upstream, create the train/validation/test split by location before this
workflow using the repository's data utilities.

To train and evaluate a model, open the notebook in Kaggle directly by clicking the respective button below.

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/alexvmt/train-and-evaluate-terainet)

1. Use [Keras Image Models](https://github.com/james77777778/keras-image-models)
2. Follow [mewc-flow](https://github.com/zaandahl/mewc-flow/blob/main/requirements.txt) and [mewc-train](https://github.com/zaandahl/mewc-train)
3. Log experiments using [Weights & Biases](https://wandb.ai/alexvmt/TeraiNet/overview)

I selected a pre-trained EfficientNetV2M with 54M parameters because it constitutes a good compromise between predictive performance, training time and model size.
The model has been trained for 46 epochs (with early stopping) with about 2,000 images per class (250 images per class in the validation set). The model has been evaluated on a separate test set with 250 images per class.

*Note: There are only ~300 tiger images on LILA BC. I didn't use them in training and validation but instead put all of them in `test2` to examine how the model would potentially generalize to tiger camera trap images from another source than the tiger training images (like it would be the case with the Nepal Tiger Trust using the model on their own images through AddaxAI, for example).*

*Note: As an extension to the underlying paper, to reduce label ambiguity in the data, images were removed if containing more than one animal detection. This was done to address cases where the image-level label did not reliably correspond to the cropped animal region, such as images containing multiple species (e.g., a bird sitting on an elephant) or incorrect crop assignments caused by generic animal detection. Although this filtering reduced the overall amount of data, it substantially improved label consistency and reduced structured supervision noise. The resulting model shows a clear performance increase, particularly for previously confusion-prone classes such as bird, rhino, elephant, and leopard, indicating that improved data quality outweighed the loss in dataset size. The main exception was the deer class, which showed a slight decline in recall and F1-score, likely because the filtering step disproportionately removed valid multi-individual or herd images that are common for deer in camera trap datasets.*

### Test set performance overview

- Accuracy: 0.935
- Precision: 0.936
- Recall: 0.935
- F1-Score: 0.935

### Test set performance per class

| Class            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| tiger            | 0.996     | 0.996  | 0.996    | 250     |
| leopard          | 0.947     | 0.936  | 0.942    | 250     |
| black_bear       | 0.914     | 0.940  | 0.927    | 250     |
| other_carnivores | 0.941     | 0.892  | 0.916    | 250     |
| deer             | 0.978     | 0.908  | 0.942    | 250     |
| wild_boar        | 0.930     | 0.956  | 0.943    | 250     |
| buffalo          | 0.885     | 0.928  | 0.906    | 250     |
| rhino            | 0.901     | 0.948  | 0.924    | 250     |
| elephant         | 0.923     | 0.916  | 0.920    | 250     |
| bird             | 0.939     | 0.928  | 0.934    | 250     |
| **micro avg**    | 0.935     | 0.935  | 0.935    | 2500    |
| **macro avg**    | 0.936     | 0.935  | 0.935    | 2500    |
| **weighted avg** | 0.936     | 0.935  | 0.935    | 2500    |

*Note: Test set performance for the tiger class is extremely high because the respective images are unrealistically good. The performance on the tiger images in `test2` is more realistic (`test2` accuracy: 0.885).*

### Test set confusion matrix

![confusion_matrix](media/confusion_matrix.png 'confusion_matrix')

*Note: The previously present confusion between certain classes is largely gone - at the slight expense of the performance on the deer class though.*

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
