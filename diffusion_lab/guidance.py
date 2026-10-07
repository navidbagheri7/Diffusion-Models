"""Classifier-free guidance and gradients of a noisy-time classifier."""

import torch
from torch.nn import functional as F


def classifier_free(model, x, t, labels=None, scale=1.0):
    """u_null + scale*(u_cond-u_null); scale 0 is unconditional, 1 conditional."""
    if labels is None:
        return model(x, t, None)
    if scale == 0:
        return model(x, t, None)
    conditional = model(x, t, labels)
    if scale == 1:
        return conditional
    unconditional = model(x, t, None)
    return unconditional + scale * (conditional - unconditional)


def classifier_gradient(classifier, x, t, labels):
    """Compute grad_x log p_phi(y|x,t), including during no_grad sampling."""
    with torch.enable_grad():
        variable = x.detach().requires_grad_(True)
        log_prob = F.log_softmax(classifier(variable, t), dim=-1)
        selected = log_prob.gather(1, labels[:, None]).sum()
        return torch.autograd.grad(selected, variable)[0].detach()


def drop_labels(labels, classes, probability, generator=None):
    if not 0 <= probability <= 1:
        raise ValueError("Label dropout must lie in [0,1]")
    mask = torch.rand(labels.shape, device=labels.device, generator=generator) < probability
    return torch.where(mask, classes, labels)


class AnalyticGMMClassifier:
    """Known noised class posterior, providing a reference for classifier guidance."""

    def __init__(self, mixture, path):
        self.mixture = mixture
        self.path = path

    def __call__(self, x, t):
        alpha, sigma = self.path.coefficients(t)
        return self.mixture.log_components(x, alpha, sigma)
