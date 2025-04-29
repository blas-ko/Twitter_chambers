# Twitter Chambers
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15303965.svg)](https://doi.org/10.5281/zenodo.15303965) [![arXiv](https://img.shields.io/badge/arXiv-2206.14501-b31b1b.svg)](https://arxiv.org/abs/2206.14501)

Code for Kolic and Aguirre-Lopez' article: "[*Quantifying the structure of controversial discussions with unsupervised methods: a look into the Twitter climate change conversation*](https://arxiv.org/abs/2206.14501)"

## Instructions
To reproduce the analysis and plots from the paper: 
1. Install the required python libraries via
> pip install -r requirements.txt
2. Download the anonymized weekly retweet networks from [Zenodo](https://doi.org/10.5281/zenodo.15303965) and paste them at `data/networks_anonymized`
3. Run every cell of [`notebooks/plots.ipynb`](https://github.com/blas-ko/Twitter_chambers/blob/main/notebooks/plots.ipynb). This notebook runs [`main.py`](https://github.com/blas-ko/Twitter_chambers/blob/main/main.py) and then creates each plot of the paper.

Additionally, check the `/example` folder and the [`chamber_example_higgs-boson.ipynb`](https://github.com/blas-ko/Twitter_chambers/blob/main/example/chamber_example_higgs-boson.ipynb) notebook for a tutorial on how to use the code with a [real retweet network](https://github.com/blas-ko/Twitter_chambers/tree/main/data/higgs_bosson_2012) about the discovery of the Higgs Bosson on 2012.

## Disclaimer
By using this code, you agree with the following points:
- The code is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations resulting from the application of the code.
- You commit to cite our paper in publications where you use or modify it.

## Contact
- **blas.kolic@uc3m.es**
- **aguirre.fabian@gmail.com**
