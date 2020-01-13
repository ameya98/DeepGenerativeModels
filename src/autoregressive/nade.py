# Neural Autoregressive Density Estimator (NADE)
# Author: Ameya Daigavane

import torch
import torch.nn as nn
import torch.distributions as dist

class NADE(nn.Module):

    # Initialize.
    def __init__(self, inp_dimensions, latent_dimensions):
        super(NADE, self).__init__()
        self.inp_dimensions = inp_dimensions
        self.latent_dimensions = latent_dimensions
        self.hidden = nn.Linear(inp_dimensions, latent_dimensions)
        self.alpha_weights = torch.rand(inp_dimensions, latent_dimensions, requires_grad=True)
        self.alpha_bias = torch.rand(inp_dimensions, requires_grad=True)

        # Helper matrix to compute prefix sums of dot-products for the forward pass.
        self.sum_matrix = torch.ones(inp_dimensions, inp_dimensions, requires_grad=False)
        for rownum, row in enumerate(self.sum_matrix):
            row[rownum:] = 0
    
    # For a given input x, obtain the mean vectors describing the Bernoulli distributions for each dimension, and each sample.
    def mean_vectors(self, x):
        # Expand each sample as a diagonal matrix.
        x_diag = torch.stack([torch.diag(x_j) for x_j in x])

        # Compute xi*Wi + bi for each dimension in each sample.
        dot_products = self.hidden(x_diag)

        # Sigmoids of prefix sums of above to get hidden activations.
        hidden_activations = torch.sigmoid(torch.matmul(self.sum_matrix, dot_products))

        # Then multiply element-wise with alpha to get mean vectors.
        return torch.sigmoid(torch.sum(torch.mul(hidden_activations, self.alpha_weights), dim=2) + self.alpha_bias)

    # Forward pass to compute log-likelihoods for each input separately.
    def forward(self, x):
        # Obtain mean vectors.
        bernoulli_means = self.mean_vectors(x)

        # Compute log-likelihoods using the mean vectors.
        log_bernoulli_means = torch.log(bernoulli_means)
        log_likelihoods = x * (log_bernoulli_means) + (1 - x) * (1 - log_bernoulli_means)

        return torch.sum(log_likelihoods, dim=1)

    # Sample.
    def sample(self, num_samples):
        samples = torch.zeros(num_samples, self.inp_dimensions)
        for sample_num in range(num_samples):
            sample = torch.zeros(self.inp_dimensions)
            for dim in range(self.inp_dimensions):
                h_dim = torch.sigmoid(self.hidden(sample))
                bernoulli_mean_dim = torch.sigmoid(self.alpha_weights[dim].dot(h_dim) + self.alpha_bias[dim])
                distribution = dist.bernoulli.Bernoulli(probs=bernoulli_mean_dim)
                sample[dim] = distribution.sample()
            samples[sample_num] = sample
        return samples
