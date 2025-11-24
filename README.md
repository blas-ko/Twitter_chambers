# From chambers to echo chambers: quantifying polarization with a second-neighbor approach applied to Twitter’s climate discussion
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15303965.svg)](https://doi.org/10.5281/zenodo.15303965)
[![DOI](https://img.shields.io/badge/DOI-10.1093/comnet/cnaf020-blue.svg)](https://doi.org/10.1093/comnet/cnaf020)
[![arXiv](https://img.shields.io/badge/arXiv-2206.14501-b31b1b.svg)](https://arxiv.org/abs/2206.14501)

Code for Kolic and Aguirre-Lopez' 2025 article: [From chambers to echo chambers: Quantifying polarization with a second-neighbor approach applied to Twitter's climate discussion](https://doi.org/10.1093/comnet/cnaf020) ([arXiv](https://arxiv.org/abs/2206.14501)), published in the *Journal of Complex Networks (2025)*.

## Instructions
To reproduce the analysis and plots from the paper: 
1. Install the required Python (`python >= 3.10`) libraries via
> pip3 install -r requirements.txt
2. Download the anonymized weekly retweet networks from [Zenodo](https://doi.org/10.5281/zenodo.15303965) and paste them at `data/networks_anonymized`
3. Run every cell of [`notebooks/plots.ipynb`](https://github.com/blas-ko/Twitter_chambers/blob/main/notebooks/plots.ipynb). This notebook runs [`main.py`](https://github.com/blas-ko/Twitter_chambers/blob/main/main.py) and then creates each plot of the paper.

Additionally, check the `/example` folder and the [`chamber_example_higgs-boson.ipynb`](https://github.com/blas-ko/Twitter_chambers/blob/main/example/chamber_example_higgs-boson.ipynb) notebook for a tutorial on how to use the code with a [real retweet network](https://github.com/blas-ko/Twitter_chambers/tree/main/data/higgs_bosson_2012) about the discovery of the Higgs Boson in 2012.

## Disclaimer
By using this code, you agree to the following points:
- The code is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations resulting from the application of the code.
- We would appreciate it if you cite our paper in publications where you use or modify it.

## Citation

### Bib
```
@article{kolic2025chambers,
  title={From chambers to echo chambers: quantifying polarization with a second-neighbor approach applied to Twitter’s climate discussion},
  author={Kolic, Blas and Aguirre-L{\'o}pez, Fabi{\'a}n and Hern{\'a}ndez-Williams, Sergio and Gardu{\~n}o-Hern{\'a}ndez, Guillermo},
  journal={Journal of Complex Networks},
  volume={13},
  number={4},
  pages={cnaf020},
  year={2025},
  publisher={Oxford University Press}
}
```

### Apa
> Kolic, B., Aguirre-López, F., Hernández-Williams, S., & Garduño-Hernández, G. (2025). From chambers to echo chambers: quantifying polarization with a second-neighbor approach applied to Twitter’s climate discussion. Journal of Complex Networks, 13(4), cnaf020.

## Contact
- **blas.kolic@uc3m.es**
- **fabian.aguirre-lopez@ladhyx.polytechnique.fr**

## License

This project is licensed under the MIT License.
