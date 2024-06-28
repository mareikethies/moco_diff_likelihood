import sys
import torch
import numpy as np
from pathlib import Path
from skimage.io import imread
from torch.utils.data import Dataset
from helper_diffusion_likelihood import setup_geometry
from skimage.metrics import structural_similarity as ssim
sys.path.append('motion_models')
from motion_models.motion_models_2d_torch import MotionModel2DTorch
sys.path.append('geometry_gradients_CT')
from geometry_gradients_CT.backprojector_fan import DifferentiableFanBeamBackprojector


device = torch.device('cuda')


class MotionCTDataset(Dataset):

    def __init__(self, data_dir, subject_ids):
        self.root_dir = Path(data_dir)
        self.ids = ['CQ500CT' + str(id) for id in subject_ids]

        self.max_rotation = 0.26  # this corresponds to 15°
        self.max_translation = 10  # [mm]

        self.images, self.metrics = self.load_data()  # load to RAM

    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, item):
        image = self.images[item]
        target = self.metrics[item]

        return image, target

    def load_data(self):
        backward_projector = DifferentiableFanBeamBackprojector.apply
        _, geometry_differentiable, proj_mat_ground_truth = setup_geometry('full')

        images = []
        metrics = []
        for subject in self.ids:
            # for all slices belonging to that subject, load the sinogram and add motion
            for slice in (self.root_dir / subject).iterdir():
                filtered_sinogram = torch.from_numpy(imread(slice / 'filtered_projections.tif')).to(device)
                # backproject with perturbed matrices
                with torch.no_grad():
                    proj_mat_perturbed = self.add_motion(proj_mat_ground_truth)
                    perturbed_reco = backward_projector(torch.squeeze(filtered_sinogram), proj_mat_perturbed,
                                                        geometry_differentiable)
                    perturbed_reco = perturbed_reco.cpu().numpy() / 175000.

                ground_truth_reco = imread(slice / 'reco.tif')

                metric = ssim(ground_truth_reco, perturbed_reco, data_range=1)

                images.append(perturbed_reco[None, :, :])
                metrics.append(metric)

        return images, metrics

    def add_motion(self, projection_matrices):
        with torch.no_grad():
            projection_matrices = torch.moveaxis(projection_matrices, 0, 2)

            std_rot = self.max_rotation * torch.rand(1)
            std_tx = self.max_translation * torch.rand(1)
            std_ty = self.max_translation * torch.rand(1)

            # set up the perturbation as required by the motion model
            motion_model = MotionModel2DTorch('spline_akima', num_nodes=10, num_projections=360)
            free_parameters = torch.zeros(motion_model.free_parameters, device=device)
            r = (torch.rand(motion_model.num_nodes) - 0.5) * min(torch.abs(torch.normal(0., std_rot)), self.max_rotation)
            tx = (torch.rand(motion_model.num_nodes) - 0.5) * min(torch.abs(torch.normal(0., std_tx)), self.max_translation)
            ty = (torch.rand(motion_model.num_nodes) - 0.5) * min(torch.abs(torch.normal(0., std_ty)), self.max_translation)
            free_parameters[0::3] = r
            free_parameters[1::3] = tx
            free_parameters[2::3] = ty

            projection_matrices_perturbed = motion_model.eval(free_parameters, projection_matrices,
                                                              return_motion_curves=False, do_zero_center=True,
                                                              is_radian=False)
            projection_matrices_perturbed = torch.moveaxis(projection_matrices_perturbed, 2, 0)

        return projection_matrices_perturbed

