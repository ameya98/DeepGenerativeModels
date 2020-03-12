#!/usr/bin/env python3

# External dependencies.
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.distributions import Uniform, MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler

# Internal dependencies.
from realnvp import MixtureDistribution, RealNVP, stackedRealNVP

if __name__ == "__main__":

    # Global properties of the data and model.
    num_classes = 2
    inp_dimensions = 2
    num_samples_per_class = 1000
    num_training_iterations = 5000
    seed = 0

    # Seed for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset.
    points, class_labels = datasets.make_moons(n_samples=num_samples_per_class*num_classes, noise=.05)
    inps_all = torch.as_tensor(StandardScaler().fit_transform(points)).float()

    inps = torch.zeros((num_classes, num_samples_per_class, inp_dimensions))
    inps[0] = inps_all[class_labels == 0]
    inps[1] = inps_all[class_labels == 1]
    
    # Class probabilities from labels.
    class_probs = F.one_hot(torch.as_tensor(class_labels)).float()

    # Learns mu and sigma parameters for RealNVP.
    mu_net = nn.Sequential(
                nn.Linear(inp_dimensions//2, 100),
                nn.LeakyReLU(),
                nn.Linear(100, inp_dimensions//2))

    sig_net = nn.Sequential(
                nn.Linear(inp_dimensions//2, 100),
                nn.LeakyReLU(),
                nn.Linear(100, inp_dimensions//2))

    # Define distributions for the latent-space encodings for each class.
    mixture = True
    if mixture:
        dist = MixtureDistribution(dists=[
            MultivariateNormal(loc=torch.ones(inp_dimensions) * -1.0, covariance_matrix=torch.eye(inp_dimensions)),
            MultivariateNormal(loc=torch.ones(inp_dimensions) * +1.0, covariance_matrix=torch.eye(inp_dimensions)),
        ], dims=inp_dimensions)
    else:
        dist = MultivariateNormal(loc=torch.zeros(inp_dimensions), covariance_matrix=torch.eye(inp_dimensions))

    # Characteristics of each RealNVP layer.
    layer_wise_dict = {
        layer_num: {
            'mu_net': deepcopy(mu_net),
            'sig_net': deepcopy(sig_net),
            'base_dist': dist,
            'dims': inp_dimensions,
            'class_condition': True,
        } for layer_num in range(3)
    }

    # Define stacked RealNVP model.
    model = stackedRealNVP(layer_wise_dict)

    # Try learning!
    optim = torch.optim.Adam(model.parameters(), lr=1e-03)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9999)
    inps_flattened = torch.flatten(inps, end_dim=1)
    
    for iteration in range(num_training_iterations):
        optim.zero_grad()

        z, log_pz, log_jacob = model(inps_flattened, class_probs)
        loss = -(log_jacob + log_pz).mean()

        if iteration % 100 == 0:
            print("Iteration %d: Loss = %0.3f" % (iteration, loss))
        
        if torch.isnan(loss):
            break

        loss.backward()
        optim.step()
        scheduler.step()

    # Latent space projection test.
    encodings = torch.zeros((num_classes, num_samples_per_class, inp_dimensions))
    for class_num in range(num_classes):
        encodings[class_num] = model(inps[class_num], class_probs[class_labels == class_num])[0].detach()

    # Inversion test.
    inps_recreated = torch.zeros((num_classes, num_samples_per_class, inp_dimensions))
    for class_num in range(num_classes):
        inps_recreated[class_num] = model.inverse(encodings[class_num]).detach()
    
    # Check, allowing 1% error.
    assert(torch.allclose(inps_recreated, inps, rtol=1e-02))
    print('Sum of L1 distances between recreated and original input: %0.4f' % torch.sum(torch.abs(inps_recreated - inps)))

    # Sample from latent distributions and invert.
    z = dist.sample([num_samples_per_class * num_classes])
    reconstructed = model.inverse(z).detach()

    # Plot samples and latent-space encodings.
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(14, 10))
    for class_num in range(num_classes):
        axs[0][0].scatter(x=inps[class_num, :, 0], y=inps[class_num, :, 1])
        axs[0][1].scatter(x=encodings[class_num, :, 0], y=encodings[class_num, :, 1])
        axs[1][0].scatter(x=z[:, 0], y=z[:, 1], c='tab:red')
        axs[1][1].scatter(x=reconstructed[:, 0], y=reconstructed[:, 1], c='tab:red')
    
    for ax in axs.flatten():
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    axs[0][0].set_title('Data Samples')
    axs[0][1].set_title('Projections into Latent Space')
    axs[1][0].set_title('Samples from Latent Distribution')
    axs[1][1].set_title('Reconstructions of Samples')

    plt.tight_layout(pad=1.)
    plt.suptitle('Stacked RealNVP', verticalalignment='top', horizontalalignment='center', y=1.)
    plt.show()
