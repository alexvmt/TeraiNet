# Tiger classification

This repository contains scripts and notebooks to classify tigers (and other species) in camera trap images,
using ML (e. g. [MegaDetector](https://github.com/agentmorris/MegaDetector)),
open source tools and data (e. g. [LILA BC](https://lila.science/))
and free compute resources (i. e. Colab and Kaggle).

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

**Sample and download images (Colab)**

1. Download image URLs and labels from LILA BC
2. For each selected species: sample and download images, create train test split if applicable
3. Copy images to Drive

*Note: Since Colab and Drive have limited capacities, one might have to further split up the process.*

**Preprocess images (Kaggle)**

1. Run [MegaDetector](https://github.com/agentmorris/MegaDetector) on all images
2. Snip images
3. Copy snipped images to Kaggle Output

*Note: Images must have been previously downloaded to Drive via Colab and then uploaded to Kaggle (zipped folder).*

## Training

- [MEWC](https://github.com/zaandahl/mewc)
- EfficientNetV2
- Kaggle

## Deployment

- HuggingFace
- [EcoAssist](https://addaxdatascience.com/ecoassist/)

Join [AI for Conservation Slack](https://beerys.github.io/#slack) and [WILDLABS](https://wildlabs.net/) if you're interested in using technology for conservation.

Feel free to reach out if you have feedback/ideas or would like to contribute/collaborate!
