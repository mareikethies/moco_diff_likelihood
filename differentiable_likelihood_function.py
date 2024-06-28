# this source code is heavily inspired and adapted from score_sde_pytorch/likelihood.py

import sys
import torch
import torch.nn as nn

sys.path.append('./score_sde_pytorch')
from models import utils as mutils


class LikelihoodODE(nn.Module):
    def __init__(self, score_model, sde, epsilon, shape):
        super().__init__()

        self.score_model = score_model
        self.sde = sde
        self.epsilon = epsilon
        self.shape = shape
        self.nfe = 0

    def drift_fn(self, model, x, t):
        """The drift function of the reverse-time SDE."""
        score_fn = mutils.get_score_fn(self.sde, model, train=False, continuous=True)
        # Probability flow ODE is a special case of Reverse SDE
        rsde = self.sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def div_fn(self, model, x, t, noise):
        return get_div_fn(lambda xx, tt: self.drift_fn(model, xx, tt))(x, t, noise)

    def forward(self, t, x):
        self.nfe += 1
        sample = x[:-self.shape[0]].reshape(self.shape)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = self.drift_fn(self.score_model, sample, vec_t).reshape(-1, )
        logp_grad = self.div_fn(self.score_model, sample, vec_t, self.epsilon).reshape(-1, )
        return torch.cat([drift, logp_grad], dim=0)


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        # add the clone because requires_gradient wil only change after the first operation on the input tensor is no_grad
        # is enabled; otherwise is_backward will be true for the first call of the forward function
        x = x.clone()
        is_backward = x.requires_grad
        # need to explicitly enable grad since evaluating the ODE with the adjoint method disables gradient tracking
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            # need to set create_graph=True, but only when in backward
            # https://discuss.pytorch.org/t/what-determines-if-torch-autograd-grad-output-has-requires-grad-true/17104
            if is_backward:
                grad_fn_eps = torch.autograd.grad(fn_eps, x, create_graph=True, retain_graph=True)[0]
            else:
                grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn