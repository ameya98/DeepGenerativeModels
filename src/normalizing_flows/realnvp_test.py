#!/usr/bin/env python3

# External dependencies.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch.distributions import Uniform, MultivariateNormal

# Internal dependencies.
from realnvp import MixtureDistribution, RealNVP, stackedRealNVP

if __name__ == "__main__":

    # Global properties of the data and model.
    num_classes = 4
    inp_dimensions = 2
    num_samples_per_class = 50
    num_training_iterations = 5000

    # Class labels and probabilities.
    class_labels = torch.arange(num_classes).repeat_interleave(num_samples_per_class)
    class_probs = F.one_hot(class_labels).float()

    # Define datapoints for each class.
    inps = torch.stack([
                torch.randn(num_samples_per_class, inp_dimensions)/10 + torch.tensor([1, 0]),
                torch.randn(num_samples_per_class, inp_dimensions)/10 + torch.tensor([0, 0]),
                torch.randn(num_samples_per_class, inp_dimensions)/10 + torch.tensor([1, 1]),
                torch.randn(num_samples_per_class, inp_dimensions)/10 + torch.tensor([0, 1])
    ])

    # Learns mu and sigma parameters for RealNVP.
    mu_net = nn.Sequential(
                nn.Linear(inp_dimensions//2, 10),
                nn.LeakyReLU(),
                nn.Linear(10, 10),
                nn.LeakyReLU(),
                nn.Linear(10, inp_dimensions//2))

    sig_net = nn.Sequential(
                nn.Linear(inp_dimensions//2, 10),
                nn.LeakyReLU(),
                nn.Linear(10, 10),
                nn.LeakyReLU(),
                nn.Linear(10, inp_dimensions//2))

    # Define distributions for the latent-space encodings for each class.
    dist = MixtureDistribution(dists=[
        MultivariateNormal(loc=torch.zeros(inp_dimensions), covariance_matrix=torch.eye(inp_dimensions)),
        MultivariateNormal(loc=torch.zeros(inp_dimensions), covariance_matrix=torch.eye(inp_dimensions)),
        MultivariateNormal(loc=torch.zeros(inp_dimensions), covariance_matrix=torch.eye(inp_dimensions)),
        MultivariateNormal(loc=torch.zeros(inp_dimensions), covariance_matrix=torch.eye(inp_dimensions)),
    ], dims=inp_dimensions)

    # Characteristics of each RealNVP layer.
    layer_wise_dict = {
        layer_num: {
            'mu_net': deepcopy(mu_net),
            'sig_net': deepcopy(sig_net),
            'base_dist': dist,
            'dims': inp_dimensions,
            'class_condition': True
        } for layer_num in range(3)
    }

    # Define stacked RealNVP model.
    model = stackedRealNVP(layer_wise_dict)

    # Latent space projection test.
    encodings = torch.zeros((num_classes, num_samples_per_class, inp_dimensions))
    for class_num in torch.arange(num_classes):
        encodings[class_num] = model(inps[class_num], class_probs[class_num * num_samples_per_class: (class_num + 1) * num_samples_per_class])[0]

    # Inversion test.
    inps_recreated = torch.zeros((num_classes, num_samples_per_class, inp_dimensions))
    for class_num in torch.arange(num_classes):
        inps_recreated[class_num] = model.inverse(encodings[class_num])
