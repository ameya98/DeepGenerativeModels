#!/usr/bin/env python3

# External dependencies.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Uniform, MultivariateNormal

# Internal dependencies.
from realnvp import MixtureDistribution, RealNVP, stackedRealNVP

if __name__ == "__main__":

    # Global properties of the data and model.
    num_classes = 4
    inp_dimensions = 10
    num_samples_per_class = 50
    num_training_iterations = 5000

    # Class labels and probabilities.
    class_labels = torch.arange(num_classes).repeat_interleave(num_samples_per_class)
    class_probs = F.one_hot(class_labels).float()

    # Define datapoints for each class.
    inps = torch.stack([
                torch.randn(num_samples_per_class, inp_dimensions)/10 + torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 0, 1]),
                torch.randn(num_samples_per_class, inp_dimensions)/10 + torch.tensor([0, 0, 1, 0, 0, 1, 0, 1, 0, 1]),
                torch.randn(num_samples_per_class, inp_dimensions)/10 + torch.tensor([1, 1, 0, 1, 1, 0, 0, 1, 0, 0]),
                torch.randn(num_samples_per_class, inp_dimensions)/10 + torch.tensor([0, 1, 1, 1, 0, 1, 1, 0, 1, 1])], dim=0)

    mu_net = nn.Sequential(
                    nn.Linear(5, 10),
                    nn.LeakyReLU(),
                    nn.Linear(10, 10),
                    nn.LeakyReLU(),
                    nn.Linear(10, 5))
    
    sig_net = nn.Sequential(
                    nn.Linear(5, 10),
                    nn.LeakyReLU(),
                    nn.Linear(10, 10),
                    nn.LeakyReLU(),
                    nn.Linear(10, 5))

    dist = MixtureDistribution(dists=[
        MultivariateNormal(loc=torch.zeros(inp_dimensions), covariance_matrix=torch.eye(inp_dimensions)),
        MultivariateNormal(loc=torch.zeros(inp_dimensions), covariance_matrix=torch.eye(inp_dimensions)),
        MultivariateNormal(loc=torch.zeros(inp_dimensions), covariance_matrix=torch.eye(inp_dimensions)),
        MultivariateNormal(loc=torch.zeros(inp_dimensions), covariance_matrix=torch.eye(inp_dimensions)),
    ], dims=inp_dimensions)

    model = RealNVP(mu_net, sig_net, dist, dims=inp_dimensions, class_condition=True)
    encodings = model(inps[0], class_probs[:num_samples_per_class])
    # Define distributions for the latent-space encodings for each class.
