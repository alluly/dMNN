# Modeling Extremes with dMNNs and Generative Models

This repository provides code to model high dimensional extreme value distributions. The repository has two algorithms: one based on a neural network architecture for representing the Pickands dependence function and another based on a generative model to represent the spectral representation. To get started, check out the two notebooks called `Estimation Example.ipynb` and `Sampling Example.ipynb`. The rest of the code provides more experiments for the proposed method in the paper ["Modeling Extremes with d-max-decreasing Neural Networks"](https://arxiv.org/abs/2102.09042).

### Background on EVT
Extreme value theory (EVT) describes the distributions of samples deep inside of the tails of a distribution. These describe, for example, appropriately scaled large magnitude events. 
In the univariate case, EVT provides simple, analytical distributions for limiting distributions of maxima/minima. 
However, in the multi-dimensional case, very few closed form examples exist for these distributions.
The preferred modeling choice comes from using copulas, which seperate the marginals from the joint dependence structure. 
The dependence structure in the multivariate extreme value case is known as the Pickands dependence function. 
This repository provides algorithms for representing this function both parametrically and stochastically. 

### CDF Estimation
Since we are working with CDFs and differentiating CDFs in high dimensions is prone to numerical errors, we consider estimation techniques specific to extreme value copulas. 
This involves computing a variable transformation such that maximum likelihood can be performed on the transformed data. 
The repository implements this method and other projections of existing nonparameteric Pickands estimators onto the space of functions parameterized by dMNNs to ensure the necessary properties are enforced.
See the example in `Estimation Example.ipynb` for more details.

### Sampling
Sampling for multivariate extreme value distributions is done using the so-called spectral measure which describes the asymptotic dependence between the covariates. 
The repository provides code for sampling by representing the spectral measure using a generative model.
See the example in `Sampling Example.ipynb` for more details.

#### Crypto and S&P Data
Since the Crypto and S&P 500 data are large, we provide scripts to download them from the source. 
Simply run the `download_cryptocmd.py` and `download_sp.py` scripts to download the data.
Make sure the cryptocmd package [1] is installed and to replace the AlphaVantage key with your own.

[1] (https://github.com/guptarohit/cryptoCMD)

### Questions and Comments
If there's anything you're interested in with respect to these algorithms, feel free to send the authors an email.

### Citation 
If you find any of this repository useful please cite the following paper:
[Modeling extremes with d-max-decreasing Neural Networks](https://arxiv.org/abs/2102.09042)

```
@misc{https://doi.org/10.48550/arxiv.2102.09042,
  doi = {10.48550/ARXIV.2102.09042},
  url = {https://arxiv.org/abs/2102.09042},
  author = {Hasan, Ali and Elkhalil, Khalil and Ng, Yuting and Pereira, Joao M. and Farsiu, Sina and Blanchet, Jose H. and Tarokh, Vahid},
  title = {Modeling Extremes with d-max-decreasing Neural Networks},
  publisher = {arXiv},
  year = {2021},
}
```

