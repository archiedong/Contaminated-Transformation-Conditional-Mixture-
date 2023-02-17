<p align="center">
  <img src = "https://user-images.githubusercontent.com/60518209/219705580-ffb94e46-e520-45ac-9ec6-58bab4196e17.png" width = "630" />
</p>

# Contaminated-Transformation-Conditional-Mixture-
We propose a contaminated transformation conditional mixture model and demonstrate on a series of simulation studies that it can effectively account for skewness and heavy tails.

## Abstract
Overparameterization is a serious concern for multivariate mixture models as it can lead to
model overfitting and, as a result, mixture order underestimation. Parsimonious modeling is
one of the most effective remedies in this context. In Gaussian mixture models, the majority
of parameters is associated with covariance matrices and parsimonious models based on factor
analyzers and spectral decomposition of dispersion parameters are the most popular in literature.
Some drawbacks of these models include the lack of flexibility in imposing different
covariance structures for individual components and limitations in modeling compact clusters.
Recently introduced conditional mixture models provide substantial flexibility in addressing
these concerns. The components of such mixtures are formulated as a product of conditional
distributions with univariate Gaussian densities being the primary choice. However, the presence
of heavy tails or skewness in any dimension can lead to fitting problems. We propose
a flexible model that is free of the above-mentioned limitations and name it a contaminated
transformation conditional mixture model and demonstrate on a series of simulation studies
that it can effectively account for skewness and heavy tails. Applications to real-life data sets
show good results and highlight the promise of the proposed model. 


## Methodology
Let $X_1, \cdots, X_n$ represents a random sample consisting of $n$ $p$-variate observations distributed according to a finite mixture model with pdf of the form,

```math
g(x; \Theta) = \sum \pi_k h(x; \vartheta_k).
```
In this expression, $K$ represents the number of components $h(x;\vartheta_k)$ in the mixture, $\vartheta_k$ stands for the component-specific parameter vector, and $\pi_k$ is the $k^{th}$ mixing proportion, subject to restrictions $\sum \pi_k = 1$ and $0 < \pi_k \le 1$. 
Popular choices of the component $h(x; \vartheta_k)$ include multivariate Gaussian, t, skew-normal, and skew-t. All these distributions include a covariance matrix with a potentially large number of parameters. As discussed, potential overparameterization is one of the major concerns in the mixture modeling framework. It can lead to model overfitting and, as a result, mixture order underestimation. Traditional parsimonious models aim at reducing the number of model parameters by considering specific structures of covariance matrices. 

Previous work has been done on an alternative parameterization with the primary attention paid to location rather than scale parameters. Note that the joint pdf $h(x;\vartheta_k)$ can be written as the product of a marginal and $p - 1$ conditional distributions $f_1(x_1; \zeta_{k1}) \prod f_j(x_j|x_1, \cdots, x_{j-1};\zeta_{kj})$, where $\zeta_{kj}$ is the parameter vector of the $j^{th}$ distribution within the $k^{th}$ component. While various forms of the pdf $f_j$ can be considered, one natural and mathmatically convenient choice is the univariate normal pdf $\phi(x_j; \bar{x}_{j}^T \beta_{kj}, \sigma_{kj}^2)$ with variance $\sigma_{kj}^2$ and linear mean function $\tilde{x}_j^T \beta_{kj}$, where $\tilde{x}_j = (1, x_1, \cdots, x_{j-1})^\top$ and $\tilde{x}_1 = 1$, and $\beta_{kj} = (\beta_{k0}, \beta_{k1}, \cdots, \beta_{k(j-1)})^\top$ is the $k$th vector of coefficients consisting of the corresponding intercept $\beta_{k0}$ and slopes $\beta_{k(j-1)}$ for $j = 2, \cdots, p$. The proposed mixture model can be written as

```math
g(x; \Theta) = \sum_{k=1} ^{K} \pi_k \prod_{j=1}^p \phi(x_j;\tilde{x}_j^\top \beta_{kj}, \sigma_{kj}^2).
```
The estimation of the parameter vector $\Theta = {\pi_1, \cdots, \pi_{K-1}, \vartheta_1, \cdots, \vartheta_K}$ is traditionally performed by means of the the expectation-maximization (EM) algorithm. At the E-step, the conditional expected value of the complete-data log-likelihood function is estimated. This expectation is traditionally referred to as the $Q$ function. At the M step, the $Q$ function is maximized with respect to $\Theta$. The procedure stops when some pre-specified convergence criterion is met, yielding the maximum likelihood estimate $\hat{\Theta}$ and estimated posterior probabilities $\hat{\tau}_{ik}$ for $i = 1, \cdots, n, k = 1, \cdots, K$. Often times the mixture order $K$ is unknown and also required to be assessed; consequently a common approach of choosing an appropriate $K$ is based on minimizing the Bayesian information criterion (BIC) over different mixture orders.
