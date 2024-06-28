import json
import torch
import numpy as np
from pathlib import Path
from skimage.io import imread
from helper_diffusion_likelihood import setup_geometry

device = torch.device('cuda')


class Data:
    def __init__(self, image_path, motion_model, amplitude_rotation=5, amplitude_translation=5):
        self.motion_model = motion_model
        self.amplitude_rotation = amplitude_rotation
        self.amplitude_translation = amplitude_translation
        self.original_volume = imread(Path(image_path) / 'original_reco.tif').astype(np.float32)
        self.reco_ground_truth = imread(Path(image_path) / f'reco.tif').astype(np.float32)
        self.projections = np.squeeze(imread(Path(image_path) / 'filtered_projections.tif').astype(np.float32)) / 175000.

        _, _, projection_matrices_circular = setup_geometry(geometry_type='full')
        self.projection_matrices_circular = projection_matrices_circular.cpu().numpy()

    def add_motion(self):
        with torch.no_grad():
            projection_matrices = torch.from_numpy(np.moveaxis(self.projection_matrices_circular, 0, 2)).to(device)

            # set up the perturbation as required by the motion model
            free_parameters = torch.zeros(self.motion_model.free_parameters, device=device)
            assert self.motion_model.free_parameters_per_node == 3, 'A motion model with 3 parameters per node is assumed.'
            r = (torch.rand(self.motion_model.num_nodes) - 0.5) * self.amplitude_rotation
            tx = (torch.rand(self.motion_model.num_nodes) - 0.5) * self.amplitude_translation
            ty = (torch.rand(self.motion_model.num_nodes) - 0.5) * self.amplitude_translation
            free_parameters[0::3] = r
            free_parameters[1::3] = tx
            free_parameters[2::3] = ty

            projection_matrices_perturbed, motion_curves = self.motion_model.eval(free_parameters,
                                                                                  projection_matrices,
                                                                                  return_motion_curves=True,
                                                                                  do_zero_center=True,
                                                                                  is_radian=False)
            projection_matrices_perturbed = np.moveaxis(projection_matrices_perturbed.cpu().numpy(), 2, 0)

        torch.cuda.empty_cache()

        return projection_matrices_perturbed.astype(np.float32), motion_curves

    def get_sample(self):
        projection_matrices_perturbed, motion_curves = self.add_motion()

        return self.projections, self.projection_matrices_circular, projection_matrices_perturbed, \
               self.reco_ground_truth, motion_curves
