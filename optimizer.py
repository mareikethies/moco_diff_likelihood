import time
import json
import torch
import logging
import numpy as np
from pathlib import Path
import scipy.linalg as la
import scipy.optimize as opt
from skimage.io import imsave
import matplotlib.pyplot as plt
from scipy.optimize import minimize

device = torch.device('cuda')
log = logging.getLogger(__name__)


class Optimizer:
    def __init__(self, sample, algorithm_choice, target_function, options, experiment_name='', sample_name=''):
        assert algorithm_choice in ['gradient_descent', 'gradient_descent_momentum'], \
            'Not a valid optimization algorithm choice.'
        self.algorithm_choice = algorithm_choice
        self.target_function = target_function
        self.options = options
        self.experiment_name = experiment_name
        self.sample_name = sample_name
        self.sample = sample
        self.projections = torch.from_numpy(sample[0]).to(device)
        self.projection_matrices_target = torch.from_numpy(sample[1]).to(device)
        self.projection_matrices_circular = torch.from_numpy(sample[2]).to(device)
        self.reco_ground_truth = torch.from_numpy(sample[3]).to(device)
        self.motion_curves_ground_truth = sample[4]
        self.num_projections = self.projections.shape[0]
        self.res = None

        self.init = np.zeros(self.target_function.motion_model.free_parameters).astype(np.float32)

    def optimize(self):
        start = time.time()
        stepsize = np.ones_like(self.init)
        stepsize[0::3] = stepsize[0::3] * self.options['stepsize_rotation']
        stepsize[1::3] = stepsize[1::3] * self.options['stepsize_translation']
        stepsize[2::3] = stepsize[2::3] * self.options['stepsize_translation']

        if self.algorithm_choice == 'gradient_descent':
            self.res = minimize(self.target_function.evaluate,
                                x0=self.init,
                                method=gradient_descent,
                                jac=True,
                                options={'maxiter': self.options['maxiter'], 'stepsize': stepsize,
                                         'lr_decay': self.options['lr_decay'], 'gtol': self.options['gtol']})

        elif self.algorithm_choice == 'gradient_descent_momentum':
            self.res = minimize(self.target_function.evaluate,
                                x0=self.init,
                                method=gradient_descent,
                                jac=True,
                                options={'maxiter': self.options['maxiter'], 'stepsize': stepsize,
                                         'lr_decay': self.options['lr_decay'], 'gtol': self.options['gtol'],
                                         'gamma': self.options['gamma']})

        self.time_needed = time.time() - start
        return self.res

    def evaluate(self, out_folder='out'):
        # optimizer results
        print(f'Experiment {self.experiment_name}')
        print(self.res.message)
        print(f'Did optimizer terminate successfully: {self.res.success}')
        print(f'Number of iterations performed by the optimizer: {self.res.nit}')
        print(f'Number of objective function evaluations: {self.res.nfev}')
        print(f'Number of Jacobian evaluations: {self.res.njev}')
        print(f'Result x: {self.res.x}')
        print(f'Result objective function: {self.res.fun}')
        print(f'Total time needed: {self.time_needed}')
        if self.target_function.profile:
            print(f'Network time needed: {self.target_function.time_network}')
            print(f'Reco time needed: {self.target_function.time_reco}')
            print(f'Backward time needed: {self.target_function.time_backward}')

        with torch.no_grad():
            reco_init, _, projection_matrices_init = self.target_function.forward_full_resolution(torch.from_numpy(self.init).to(device))
            reco_out, motion_curves, projection_matrices_out = self.target_function.forward_full_resolution(torch.from_numpy(self.res.x).to(device))

        # create folder
        out_folder = Path(out_folder)
        out_folder = out_folder / self.algorithm_choice / self.target_function.choice / self.experiment_name / self.sample_name
        file_prefix = out_folder / f'{self.experiment_name}'

        if not out_folder.is_dir():
            out_folder.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.plot(self.target_function.losses, 'b-')
        plt.xlabel('Objective function calls')
        plt.ylabel('Objective function value')
        plt.tight_layout()
        plt.savefig(f'{file_prefix}_objective_function.png')

        plt.figure(figsize=(12, 3))
        plt.subplot(131)
        plt.plot(self.motion_curves_ground_truth[0][0].cpu().numpy(), label='gt')
        plt.plot(-motion_curves[0][0].cpu().numpy(), label='opt')
        plt.plot(self.motion_curves_ground_truth[1][0].cpu().numpy(), self.motion_curves_ground_truth[2][0].cpu().numpy(), 'x')
        plt.plot(motion_curves[1][0].cpu().numpy(), -motion_curves[2][0].cpu().numpy(), 'x')
        plt.title('Rotation [deg]')

        plt.subplot(132)
        plt.plot(self.motion_curves_ground_truth[0][1].cpu().numpy(), label='gt')
        plt.plot(-motion_curves[0][1].cpu().numpy(), label='opt')
        plt.plot(self.motion_curves_ground_truth[1][1].cpu().numpy(), self.motion_curves_ground_truth[2][1].cpu().numpy(), 'x')
        plt.plot(motion_curves[1][1].cpu().numpy(), -motion_curves[2][1].cpu().numpy(), 'x')
        plt.title('Translation x [mm]')

        plt.subplot(133)
        plt.plot(self.motion_curves_ground_truth[0][2].cpu().numpy(), label='gt')
        plt.plot(-motion_curves[0][2].cpu().numpy(), label='opt')
        plt.plot(self.motion_curves_ground_truth[1][2].cpu().numpy(), self.motion_curves_ground_truth[2][2].cpu().numpy(), 'x')
        plt.plot(motion_curves[1][2].cpu().numpy(), -motion_curves[2][2].cpu().numpy(), 'x')
        plt.title('Translation y [mm]')
        plt.legend()

        plt.savefig(f'{file_prefix}_motion_curves.png')

        plt.figure()
        plt.subplot(131)
        plt.imshow(reco_init.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.title('Perturbed')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(reco_out.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.title('Recovered')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(self.reco_ground_truth.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.title('Ground truth')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{file_prefix}_reconstructions.png')

        # close figures to free memory
        plt.close()

        # store results in dict
        results = {}
        results['nit'] = self.res.nit
        results['nfev'] = self.res.nfev
        results['njev'] = self.res.njev
        results['fun'] = float(self.res.fun)
        results['time'] = self.time_needed
        results['x'] = list(self.res.x.astype(np.float64))
        results['motion_r_init'] = list(self.motion_curves_ground_truth[0][0].cpu().numpy().astype(np.float64))
        results['motion_tx_init'] = list(self.motion_curves_ground_truth[0][1].cpu().numpy().astype(np.float64))
        results['motion_ty_init'] = list(self.motion_curves_ground_truth[0][2].cpu().numpy().astype(np.float64))
        results['motion_r_recovered'] = list(motion_curves[0][0].cpu().numpy().astype(np.float64))
        results['motion_tx_recovered'] = list(motion_curves[0][1].cpu().numpy().astype(np.float64))
        results['motion_ty_recovered'] = list(motion_curves[0][2].cpu().numpy().astype(np.float64))
        results['loss'] = self.target_function.losses

        json.dump(results, open(f'{file_prefix}_result.json', 'w'), indent=1)
        imsave(f'{file_prefix}_reco_init.tif', reco_init.cpu().numpy())
        imsave(f'{file_prefix}_reco_out.tif', reco_out.cpu().numpy())
        imsave(f'{file_prefix}_reco_target.tif', self.reco_ground_truth.cpu().numpy())

        np.save(f'{file_prefix}_projection_matrices_init.npy', projection_matrices_init.cpu().numpy())
        np.save(f'{file_prefix}_projection_matrices_out.npy', projection_matrices_out.cpu().numpy())
        np.save(f'{file_prefix}_projection_matrices_target.npy', self.projection_matrices_target.cpu().numpy())

        return file_prefix


def gradient_descent(fun, x0, jac, args, maxfev=None, stepsize=1e-5, lr_decay=1., maxiter=500, gtol=1, callback=None,
                     **options):
    log.info('Gradient descent started.')
    bestx = x0
    # bestf = fun(x0, *args)
    funcalls = 1
    niter = 0
    improved = True
    stop = False

    while improved and not stop and niter < maxiter:
        niter += 1
        grad = jac(bestx, *args)
        # print(la.norm(grad))
        if la.norm(grad) < gtol:
            improved = False
        step = stepsize * grad
        stepsize = lr_decay * stepsize
        bestx = bestx - step

        bestf = fun(bestx, *args)
        funcalls += 1

        print(f'{niter}: {bestf:.5f}')

        if callback is not None:
            callback(bestx)
        if maxfev is not None and funcalls >= maxfev:
            stop = True
            break

    return opt.OptimizeResult(fun=bestf, x=bestx, nit=niter, nfev=funcalls, njev=funcalls, success=(niter > 1),
                              message='')


def gradient_descent_momentum(fun, x0, jac, args, maxfev=None, stepsize=1e-5, lr_decay=1., maxiter=500, gtol=1,
                              gamma=0.5, callback=None, **options):
    log.info('Gradient descent started.')
    bestx = x0
    # bestf = fun(x0, *args)
    momentum = np.zeros_like(x0)
    funcalls = 1
    niter = 0
    improved = True
    stop = False

    while improved and not stop and niter < maxiter:
        niter += 1
        grad = jac(bestx, *args)

        if la.norm(grad) < gtol:
            improved = False

        # compute momentum
        momentum = gamma * momentum
        step = momentum + stepsize * grad
        stepsize = lr_decay * stepsize
        bestx = bestx - step

        bestf = fun(bestx, *args)
        funcalls += 1

        print(f'{niter}: {bestf:.5f}')

        if callback is not None:
            callback(bestx)
        if maxfev is not None and funcalls >= maxfev:
            stop = True
            break

    return opt.OptimizeResult(fun=bestf, x=bestx, nit=niter, nfev=funcalls, njev=funcalls, success=(niter > 1),
                              message='')
