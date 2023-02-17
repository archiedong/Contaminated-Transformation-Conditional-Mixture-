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

Previous work has been done on an alternative parameterization with the primary attention paid to location rather than scale parameters. Note that the joint pdf $h(x;\vartheta_k)$ can be written as the product of a marginal and $p - 1$ conditional distributions $f_1(x_1; \zeta_{k1}) \prod f_j(x_j|x_1, \cdots, x_{j-1};\zeta_{kj})$, where $\zeta_{kj}$ is the parameter vector of the $j^{th}$ distribution within the $k^{th}$ component. While various forms of the pdf $f_j$ can be considered, one natural and mathmatically convenient choice is the univariate normal pdf. The proposed mixture model can be written as

```math
g(x; \Theta) = \sum_{k=1} ^{K} \pi_k \prod_{j=1}^p \phi(x_j;\tilde{x}_j^\top \beta_{kj}, \sigma_{kj}^2).
```
The estimation of the parameter vector $\Theta = {\pi_1, \cdots, \pi_{K-1}, \vartheta_1, \cdots, \vartheta_K}$ is traditionally performed by means of the the expectation-maximization (EM) algorithm. At the E-step, the conditional expected value of the complete-data log-likelihood function is estimated. This expectation is traditionally referred to as the $Q$ function. At the M step, the $Q$ function is maximized with respect to $\Theta$. The procedure stops when some pre-specified convergence criterion is met, yielding the maximum likelihood estimate $\hat{\Theta}$ and estimated posterior probabilities $\hat{\tau}_{ik}$ for $i = 1, \cdots, n, k = 1, \cdots, K$. Often times the mixture order $K$ is unknown and also required to be assessed; consequently a common approach of choosing an appropriate $K$ is based on minimizing the Bayesian information criterion (BIC) over different mixture orders.

With the Yeo and Johnson transformation in Equation~\ref{eq:Yeo}), the joint density function in Equation becomes 
```math
\begin{split}
f(x;\theta)  =& [\delta \phi(\mathcal{T}(x; \lambda);\mu, \Sigma) + (1 - \delta) \phi(\mathcal{T}(x; \lambda);\mu, \alpha \Sigma)]  J_{\mathcal{T}}(x;\lambda) \\ 
& = [\delta \prod_{j=1}^p \phi(\mathcal{T}(x_{j}; \lambda_j);\{\hat{x}_{j}^m\}^\top \beta_{j}, \sigma^2_j) + (1 - \delta) \prod_{j=1}^p \phi(\mathcal{T}(x_{j}; \lambda_j);\{\hat{x}_{j}^m\}^\top \beta_{j}, \alpha \sigma^2_j)] \prod_{j = 1}^P J_{\mathcal{T}}(x_j;\lambda_j) \\ 
& = [\delta \phi_1 (\mathcal{T}(x_1; \lambda_1);\{\hat{x}_1^m\}^\top \beta_1, \sigma_1^2) \cdots \phi_p(\mathcal{T}(x_p; \lambda_p); \{\hat{x}_p^m\}^\top \beta_p, \sigma_p^2) \\ 
&+ (1 - \delta) \phi_1 (\mathcal{T}(x_1; \lambda_1); \{\hat{x}_1^m\}^\top \beta_1, \alpha \sigma_1^2) \cdots \phi_p(\mathcal{T}(x_p; \lambda_p); \{\hat{x}_p^m\}^\top \beta_p, \alpha \sigma_p^2)] 
\prod_{j = 1}^P J_{\mathcal{T}}(x_j;\lambda_j),
\end{split}
```
Assuming all membership labels $Z_i$, $i = 1, \ldots, n$ are known, i.e. the complete data is given by $(X_i, Z_i)$, the corresponding complete-data likelihood function can be then rewritten as

```math
L(\Psi) = \prod_{i = 1}^n \prod_{k = 1}^K \prod_{w = 1}^2 [ \pi_k [ \delta_{kw} \phi ( \mathcal{T} ( x_i; \lambda_{ jk } ); \mu_k, \Sigma_{kw} ) J_{ \mathcal{T} } ( x_i; \lambda_{ jk } ) ]^{ I(W_i = w) }]^{I(Z_i = k) },
```
where $I(Z_i = k) = 1$ if $Z_i$ belongs to the $k$th component and 0 otherwise; similarly $I(W_i = w)$ indicates if the ith observation from the kth component is contaminated. At the E-step of the algorithm, the conditional expectation of the complete-data log-likelihood function obtained from Eq requires updating posterior probabilities according to the following expressions:

```math
\ddot{\tau}_{ik} = \frac{\dot{\pi}_k [\dot{\delta}_k \prod \phi(\mathcal{T}(x_{ij}; \dot{\lambda}_{jk}); \{\tilde{x}_{ij}^m\}^\top \dot{\beta}_{jk}, \dot{\sigma}_{jk}^2) + (1 - \dot{\delta}_k) \prod \phi(\mathcal{T}(x_{ij}; \dot{\lambda}_{jk});\{\tilde{x}_{ij}^m\}^\top \dot{\beta}_{jk}, \dot{\alpha}_k \dot{\sigma}_{jk}^2)]  \prod J_{\mathcal{T}}(x_{ij};\lambda_{jk})} {\sum_{r=1}^K \dot{\pi}_k [\dot{\delta}_r \prod \phi(\mathcal{T}(x_{ij}; \dot{\lambda}_{jr});\{\tilde{x}_{ij}^m\}^\top \dot{\beta}_{jr}, \dot{\sigma}_{jr}^2) + (1 - \dot{\delta}_r) \prod \phi(\mathcal{T}(x_{ij}; \dot{\lambda}_{jr});\{\tilde{x}_{ij}^m\}^\top \dot{\beta}_{jr}, \dot{\alpha}_r \dot{\sigma}_{jr}^2)]  \prod J_{\mathcal{T}}(x_{ij};\lambda_{jr})}
```
and

```math
\ddot{\nu}_{i|k} = \frac{\dot{\delta}_k \prod \phi(\mathcal{T}(x_{ij}; \dot{\lambda}_{jk});\{\tilde{x}_{ij}^m\}^\top \dot{\beta}_{jk}, \dot{\sigma}_{jk}^2)} { \dot{\delta}_k \prod \phi(\mathcal{T}(x_{ij}; \dot{\lambda}_{jk});\{\tilde{x}_{ij}^m\}^\top \dot{\beta}_{jk}, \dot{\sigma}_{jk}^2) + (1 - \dot{\delta}_k) \prod \phi(\mathcal{T}(x_{ij}; \dot{\lambda}_{jk});\{\tilde{x}_{ij}^m\}^\top \dot{\beta}_{jk}, \dot{\alpha}_k \dot{\sigma}_{jk}^2)},
```
where $\tau_{ik}$ is the probability that $x_i$ originates from the $k^{th}$ mixture component and $\nu_{i|k}$ is the probability that $x_i$ belongs to the primary distribution within the $k^{th}$ component. In other words, $\nu_{i|k}$ is the probability that $x_i$ is not contanimated in the $k^{th}$ component. One dot and two dots on the top of parameters stand for estimates at the previous and current iterations, respectively. Therefore, the Q function, takes the following form:
