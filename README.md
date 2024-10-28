# Tiger detection

This repository contains tools to detect and classify tigers in camera trap images,
using public camera trap images from LILA BC and free GPU resources in Google Colab.

![tiger](anno_1440.jpg 'tiger')

*Credentials: LILA BC, MegaDetector, own illustration.*


## Motivation

- tigers are an endangered species
- there is no open and easy way for ecologists/researchers/NGOs
to detect and classify their camera trap images with regard to tigers
- ML and open data/tools can help reduce the amount of manual labor
when sifting through large amounts of camera trap data, looking for the needle in the haystack

## Data

**Data sources**

- [LILA BC](https://lila.science/)
- amur tiger re-identification [challenge](https://cvwc2019.github.io/challenge.html) at CVWC 2019

**Data preparation**

1. Define relevant species classes (tiger, other mamals, birds etc.)
2. Sample n images per class randomly from LILA BC (e. g. from the last i years)
3. Download sampled images and check for errors/inconsistencies
4. Run images through [MegaDetector](https://github.com/agentmorris/MegaDetector) to get bounding boxes (and filter out empty images, vehicles and people if any)
5. Use mewc-snip to crop images
6. Build train, val and test sets

## Training

- [MEWC](https://github.com/zaandahl/mewc)
- EfficientNetV2
- Google Colab

## Deployment

- HuggingFace
- [EcoAssist](https://addaxdatascience.com/ecoassist/)

Join [AI for Conservation Slack](https://beerys.github.io/#slack) and [WILDLABS](https://wildlabs.net/) if you're interested in using technology for conservation.

Feel free to reach out if you have feedback/ideas or would like to contribute/collaborate!
