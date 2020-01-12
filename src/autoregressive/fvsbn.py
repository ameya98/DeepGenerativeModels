# Fully Visible Belief Networks
# Author: Ameya Daigavane

import torch
import torch.nn as nn

class FVSBN(nn.Module):

    # Initialize.
    def __init__(self, inp_dimensions):
        super(FVSBN, self).__init__()
        self.inp_dimensions = inp_dimensions
        self.linear = nn.Linear(inp_dimensions, inp_dimensions)

        # Fix weights as 0 for indices >= row for each row.
        for row, weights in enumerate(self.linear.weight.data):
            weights[row: ].data.fill_(0)
        
    # Forward pass to compute log-likelihoods for each input separately.
    def forward(self, x):
        bernoulli_means = torch.sigmoid(self.linear(x))
        log_likelihoods = torch.log(bernoulli_means)
        return torch.sum(log_likelihoods, dim=1)

    # Do not update weights for indices >= row for each row.
    def zero_grad_for_extra_weights(self):
        for row, grads in enumerate(self.linear.weight.grad):
            grads[row: ] = 0
