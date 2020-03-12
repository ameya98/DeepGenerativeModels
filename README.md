# Deep Generative Models
My PyTorch implementations of deep generative models.

## Autoregressive Models
### Fully Visible Sigmoidal Belief Network (FVSBN)
<p align="center">
    <img src="img/fvsbn.png"></img>
</p>

### Neural Autoregressive Distribution Estimator (NADE)
<p align="center">
    <img src="img/nade.png"></img>
</p>

## Flow Models
### Real-Valued Non-Volume Preserving Flows (RealNVP)
Latent Distribution: Zero-Mean Unit-Variance Gaussian
<p align="center">
    <img src="src/realnvp_original.png"></img>
</p>

Latent Distribution: Mixture of Unit-Variance Gaussians Centered at [-1, -1] (Class 0) and [1, 1] (Class 1).
<p align="center">
    <img src="src/realnvp_class_conditioned.png"></img>
</p>
