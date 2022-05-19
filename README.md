# Modeling Extremes with dMNNs and Generative Models

This repository provides code to model high dimensional extreme value distributions. The repository has two algorithms: one based on a neural network architecture for representing the Pickands dependence function and another based on a generative model to represent the spectral representation. To get started, check out the two notebooks called `example_estimate.ipynb` and `example_sample.ipynb`. The rest of the code provides more experiments for the proposed method in the paper ["Modeling Extremes with d-max-decreasing Neural Networks"](https://arxiv.org/abs/2102.09042).

### Background on EVT
Extreme value theory (EVT) describes the distributions of samples deep inside of the tails of a distribution. These describe, for example, very large magnitude events. 
In the univariate case, EVT provides simple, analytical distributions for limiting distributions of maxima/minima. 
However, in the multi-dimensional case, very few closed form examples exist for these distributions.
The preferred modeling choice comes from using copulas, which seperate the marginals from the joint dependence structure. 
The dependence structure in the multivariate extreme value case is known as the Pickands dependence function. 
This repository provides algorithms for representing this function both parametrically and stochastically. 

## Estimation

## Sampling

#### Crypto and S&P Data
Since the crypto and sp 500 data are large, we provide scripts to download them from the source. 
Simply run the download_cryptocmd.py and download_sp.py scripts to download the data.
Make sure the cryptocmd package [1] is installed and to replace the AlphaVantage key with your own.

[1] github.com/guptarohit/crpytoCMD

### Questions and Comments
If there's anything you're interested in seeing with respect to these algorithms, feel free to send the authors an email.

### Citation 
If you find any of this repository useful please cite the following paper:
[Modeling extremes with d-max-decreasing Neural Networks](https://arxiv.org/abs/2102.09042)

@misc{https://doi.org/10.48550/arxiv.2102.09042,
  doi = {10.48550/ARXIV.2102.09042},
  url = {https://arxiv.org/abs/2102.09042},
  author = {Hasan, Ali and Elkhalil, Khalil and Ng, Yuting and Pereira, Joao M. and Farsiu, Sina and Blanchet, Jose H. and Tarokh, Vahid},
  title = {Modeling Extremes with d-max-decreasing Neural Networks},
  publisher = {arXiv},
  year = {2021},
  copyright = {Creative Commons Attribution 4.0 International}
}

