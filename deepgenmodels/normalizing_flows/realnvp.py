# Real Non-Volume Preserving Flows (Original and Class-Conditioned Formulations)
# Author: Ameya Daigavane with help from https://gebob19.github.io/normalizing-flows/

import torch
import torch.nn as nn
import numpy as np

# Real Non-Volume Preserving Flows.
class RealNVP(nn.Module):
    def __init__(self, mu_net, sig_net, base_dist, dims, class_condition=False):
        super().__init__()
        self.dims = dims
        self.split_dims = dims // 2

        # Both must map a vector of dimension floor(dims/2) to ceil(dims/2).
        self.mu_net = mu_net
        self.sig_net = sig_net

        # Distribution for the latent space.
        self.base_dist = base_dist

        # Whether to condition on classes or not.
        self.class_condition = class_condition

        # Permutation, with its inverse.
        self.permutation = torch.randperm(dims)
        self.inv_permutation = torch.zeros(dims).long()
        self.inv_permutation[self.permutation] = torch.arange(dims)

    # Compute latent-space encodings, along with log-likelihoods and log-Jacobian of transformation.
    def forward(self, x, class_probs=None):
        if self.class_condition and class_probs is None:
            raise ValueError('Parameter class_probs not specified.')

        # Apply permutation.
        x = x[:, self.permutation]

        # Split into pieces.
        x1, x2 = x[:, :self.split_dims], x[:, self.split_dims:] 

        # Apply transformation.
        z1 = x1
        sig = self.sig_net(x1)
        mu = self.mu_net(x1)
        z2 = x2 * torch.exp(sig) + mu
        
        # Join pieces.
        z_hat = torch.cat([z1, z2], dim=-1)

        # Compute log-likelihood. If class-conditioned, weight by the individual class probs.
        if self.class_condition:
            log_pzs_classwise = self.base_dist.log_probs_classwise(z_hat)
            pzs_classwise = torch.exp(log_pzs_classwise)
            log_pz = torch.log(torch.sum(torch.mul(pzs_classwise, class_probs), dim=1))
        else:
            log_pz = self.base_dist.log_prob(z_hat) 

        # Compute log-Jacobian of transformation.
        log_jacob = sig.sum(-1)
        
        return z_hat, log_pz, log_jacob
    
    # Compute inverse of given latent-space encoding.
    def inverse(self, z):
        # Split again.
        z1, z2 = z[:, :self.split_dims], z[:, self.split_dims:] 
        
        # Compute inverse transformation.
        x1 = z1
        x2 = (z2 - self.mu_net(z1)) * torch.exp(-self.sig_net(z1))
        
        # Join pieces.
        x = torch.cat([x1, x2], -1)

        # Apply inverse permutation.
        x = x[:, self.inv_permutation]

        return x


# Stack of RealNVPs.
class RealNVPStacked(nn.Module):
    def __init__(self, layer_wise_dict):
        super().__init__()
        self.bijectors = nn.ModuleList([
            RealNVP(**layer_wise_params) for layer_wise_params in layer_wise_dict.values()
        ])
        
    # Pass through each RealNVP one-by-one.
    def forward(self, x, class_probs=None):
        log_jacobs = []
        
        for bijector in self.bijectors:
            x, log_pz, lj = bijector(x, class_probs)
            log_jacobs.append(lj)
        
        return x, log_pz, sum(log_jacobs)
    
    # Invert through each RealNVP one-by-one, starting from the end.
    def inverse(self, z):
        for bijector in reversed(self.bijectors):
            z = bijector.inverse(z)
        return z
