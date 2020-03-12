# Mixture distributions for modelling multiple classes, at once.
# Author: Ameya Daigavane

class MixtureDistribution:
    def __init__(self, dims, dists, weights=None):
        self.dims = dims
        self.n_components = len(dists)
        self.dists = dists

        if weights is not None:
            if len(weights) != self.n_components:
                raise ValueError('Size of weights %d does not match with number of distributions %d.' % (len(weights, self.n_components)))
            self.weights = weights
        else:
            self.weights = torch.ones(self.n_components) / self.n_components

    def log_prob(self, x):
        probs = torch.stack([torch.exp(dist.log_prob(x)) for dist in self.dists], dim=1)
        return torch.log(torch.sum(self.weights * probs, dim=1))
    
    def log_probs_classwise(self, x):
        return torch.stack([dist.log_prob(x) for dist in self.dists], dim=1)
        
    def sample(self, sample_shape):
        n_samples = sample_shape[0]
        samples = torch.zeros((n_samples, self.dims))
        dist_indices = np.random.choice(np.arange(self.n_components), size=n_samples, p=self.weights)
        for index, dist_index in enumerate(dist_indices):
            samples[index] = self.dists[dist_index].sample()
        return samples