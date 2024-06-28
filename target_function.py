import sys
import torch
import torch.nn as nn
from time import perf_counter
from autofocus_model import AutofocusModel
from torchdiffeq import odeint_adjoint as odeint
from helper_diffusion_likelihood import setup_geometry
from differentiable_likelihood_function import LikelihoodODE
from compute_likelihood import initialize_score_model, initialize_sde
sys.path.append('geometry_gradients_CT')
from geometry_gradients_CT.backprojector_fan import DifferentiableFanBeamBackprojector


class DummyScoreModel(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x, time_cond):
        return 1e-5 * x


device = torch.device('cuda')


class TargetFunction:
    def __init__(self, choice, config, projections, motion_model, projection_matrices_circular,
                 ground_truth_reconstruction):
        assert choice in ['diffusion_likelihood', 'mse_supervised', 'autofocus'], 'Unknown target function choice.'
        self.choice = choice
        self.method = config.target_function.method
        self.adjoint_method = config.target_function.adjoint_method
        self.adjoint_rtol = config.target_function.adjoint_rtol
        self.adjoint_atol = config.target_function.adjoint_atol
        self.motion_model = motion_model
        self.projections = torch.from_numpy(projections).to(device)
        self.initial_projection_matrices = torch.from_numpy(projection_matrices_circular).to(device)
        self.backprojector = DifferentiableFanBeamBackprojector.apply
        _, self.geometry_full, _ = setup_geometry(geometry_type='full')
        self.ground_truth_reconstruction = torch.from_numpy(ground_truth_reconstruction).to(device)
        self.losses = []
        self.profile = config.target_function.profile
        self.time_reco = 0
        self.time_network = 0
        self.time_backward = 0
        self.N = 0
        self.eps = 1e-5
        self.log2 = torch.log(torch.tensor(2))

        if self.choice == 'diffusion_likelihood':
            self.score_model = initialize_score_model(config, config.target_function.checkpoint_path)
            self.sde = initialize_sde(config)

            self.shape = (1, 1, 256, 256)
            self.N = 65536
            self.epsilon = (torch.randint(size=self.shape, low=0, high=2).float() * 2 - 1.).to(device)  # Rademacher

            self.ode_func = LikelihoodODE(self.score_model, self.sde, self.epsilon, shape=self.shape)
            self.t = torch.tensor([self.eps, self.sde.T], device=device)
            self.options = {'step_size': config.target_function.stepsize}
            self.adjoint_options = {'step_size': config.target_function.adjoint_stepsize, 'norm': 'seminorm'}

        elif self.choice == 'autofocus':
            self.autofocus_model = AutofocusModel.load_from_checkpoint(
                './checkpoints/epoch=1947-step=2637592.ckpt').to(device)
            self.autofocus_model.eval()
            self.autofocus_model.freeze()

    def evaluate(self, free_parameters):
        self.free_parameters = torch.from_numpy(free_parameters).to(device)
        self.N += 1
        self.free_parameters.requires_grad = True
        reco, motion_curves, _ = self.forward(self.free_parameters)
        loss = self.compute_loss(reco)
        return loss

    def forward(self, free_parameters):
        start_reco = perf_counter()

        initial_projection_matrices = torch.moveaxis(self.initial_projection_matrices, 0, 2)
        updated_projection_matrices, motion_curves = self.motion_model.eval(free_parameters,
                                                                            initial_projection_matrices,
                                                                            return_motion_curves=True,
                                                                            do_zero_center=True,
                                                                            is_radian=False)

        updated_projection_matrices = torch.moveaxis(updated_projection_matrices, 2, 0)
        updated_projection_matrices_ = updated_projection_matrices / torch.unsqueeze(torch.unsqueeze(updated_projection_matrices[:, 1, 2], 1), 2)

        reconstruction = self.backprojector(self.projections, updated_projection_matrices_, self.geometry_full)

        if self.profile:
            torch.cuda.synchronize()
            self.time_reco += perf_counter() - start_reco

        return reconstruction, motion_curves, updated_projection_matrices_

    def compute_loss(self, reconstruction):
        start_network = perf_counter()

        loss = None
        if self.choice == 'diffusion_likelihood':
            reconstruction = torch.unsqueeze(torch.unsqueeze(reconstruction, dim=0), dim=0)
            loss = self.likelihood_fn(reconstruction)
            # self.evaluate_nfe()

        elif self.choice == 'mse_supervised':
            loss = torch.mean(torch.square(reconstruction - self.ground_truth_reconstruction))

        elif self.choice == 'autofocus':
            reconstruction = torch.unsqueeze(torch.unsqueeze(reconstruction, dim=0), dim=0)
            loss = 1. - self.autofocus_model(reconstruction)

        if self.profile:
            torch.cuda.synchronize()
            self.time_network += perf_counter() - start_network

        start_backward = perf_counter()

        loss.backward()

        # self.evaluate_nfe()

        grad = self.free_parameters.grad.detach().cpu().numpy()

        if self.profile:
            torch.cuda.synchronize()
            self.time_backward += perf_counter() - start_backward

        self.losses.append(float(loss.item()))
        torch.cuda.empty_cache()  # not quite sure why I need this; otherwise does not run locally on 8GB GPU

        return float(loss), grad

    def evaluate_nfe(self):
        print(self.ode_func.nfe)
        self.ode_func.nfe = 0

    def likelihood_fn(self, data):
        shape = data.shape
        init = torch.cat([data.reshape(-1, ), torch.zeros((shape[0]), device=data.device)], dim=0)
        solution = odeint(self.ode_func, init, self.t, method=self.method, options=self.options,
                          adjoint_options=self.adjoint_options, adjoint_rtol=self.adjoint_rtol,
                          adjoint_atol=self.adjoint_atol, adjoint_method=self.adjoint_method)
        zp = solution[-1, :]  # select last time step
        z = zp[:-shape[0]].reshape(shape)
        delta_logp = zp[-shape[0]:].reshape((shape[0],))
        prior_logp = self.sde.prior_logp(z)
        bpd = -(prior_logp + delta_logp) / self.log2
        bpd = bpd / self.N
        bpd = bpd + 8
        return bpd

    def forward_full_resolution(self, free_parameters):
        initial_projection_matrices = torch.moveaxis(self.initial_projection_matrices, 0, 2)
        updated_projection_matrices, motion_curves = self.motion_model.eval(free_parameters,
                                                                            initial_projection_matrices,
                                                                            return_motion_curves=True,
                                                                            do_zero_center=True,
                                                                            is_radian=False)
        updated_projection_matrices = torch.moveaxis(updated_projection_matrices, 2, 0)
        updated_projection_matrices_ = updated_projection_matrices / torch.unsqueeze(
            torch.unsqueeze(updated_projection_matrices[:, 1, 2], 1), 2)

        reconstruction = self.backprojector(self.projections, updated_projection_matrices_, self.geometry_full)

        return reconstruction, motion_curves, updated_projection_matrices_
