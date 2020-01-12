#!/usr/bin/env python3

# External dependencies.
import torch
import torch.optim as optim
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Internal dependencies.
from fvsbn import FVSBN

# Sample classification with artificial 3D data.
if __name__ == "__main__":

    # Seed for reproducibility.
    torch.manual_seed(0)

    # Global properties of the data and model.
    num_classes = 4
    inp_dimensions = 10
    num_samples_per_class = 50
    num_training_iterations = 500

    # Class labels.
    classes = torch.cat([
                torch.full((num_samples_per_class,), 0),
                torch.full((num_samples_per_class,), 1),
                torch.full((num_samples_per_class,), 2),
                torch.full((num_samples_per_class,), 3)], dim=0)

    # Define datapoints for each class.
    inps = torch.stack([
                torch.randn(num_samples_per_class, inp_dimensions)/10 + torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 0, 1]),
                torch.randn(num_samples_per_class, inp_dimensions)/10 + torch.tensor([0, 0, 1, 0, 0, 1, 0, 1, 0, 1]),
                torch.randn(num_samples_per_class, inp_dimensions)/10 + torch.tensor([1, 1, 0, 1, 1, 0, 0, 1, 0, 0]),
                torch.randn(num_samples_per_class, inp_dimensions)/10 + torch.tensor([0, 1, 1, 1, 0, 1, 1, 0, 1, 1])], dim=0)

    # Define one model per class.
    models = [FVSBN(inp_dimensions) for _ in range(num_classes)]

    # Train each model one by one.
    for inp, model in zip(inps, models):
        
        # Optimization scheme.
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        for _ in range(num_training_iterations):
            
            # Zero out previous gradients.
            model.zero_grad()

            # Compute log-likehoods per sample.
            log_likelihoods = model(inp)

            # Negative mean over all samples, because we're minimizing with SGD instead of maximizing.
            negative_mean_log_likehoods = -torch.mean(log_likelihoods)
            
            # Compute gradients.
            negative_mean_log_likehoods.backward()

            # Zero out gradients for the weights we don't have.
            model.zero_grad_for_extra_weights()

            # Update weights.
            optimizer.step()

    # Label datapoints by the model that gave it the highest likelihood. 
    inps_flattened = torch.flatten(inps, start_dim=0, end_dim=1)
    predicted_log_likelihoods = torch.stack([model(inps_flattened) for model in models], dim=1)
    predicted_classes = torch.argmax(predicted_log_likelihoods, dim=1)
    
    # Scatterplots of classification.
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5), subplot_kw=dict(projection='3d'))
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_zlabel('z')
    axs[0].scatter(inps_flattened[:, 0], inps_flattened[:, 1], inps_flattened[:, 2], c=classes.numpy(), cmap='Set1')
    axs[0].set_title('True Labels', y=1.05)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_zlabel('z')
    axs[1].scatter(inps_flattened[:, 0], inps_flattened[:, 1], inps_flattened[:, 2], c=predicted_classes.numpy(), cmap='Set1')
    axs[1].set_title('FVSBN Predicted Labels', y=1.05)
    plt.tight_layout()
    plt.show()

    # Scatterplots of generated samples.
    samples = torch.cat([model.sample(num_samples_per_class) for model in models], dim=0)
    fig, axs = plt.subplots(ncols=1, figsize=(5, 5), subplot_kw=dict(projection='3d'))
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.set_zlabel('z')
    axs.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=classes.numpy(), cmap='Set1')
    axs.set_title('Samples', y=1.05)
    plt.tight_layout()
    plt.show()



    