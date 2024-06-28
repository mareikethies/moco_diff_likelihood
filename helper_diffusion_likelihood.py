import sys
import torch
import numpy as np
from reconstruction_settings import *
sys.path.append('motion_models')
from motion_models.motion_models_2d_torch import MotionModel2DTorch
from geometry_gradients_CT.geometry import Geometry as GeometryDifferentiable
from pyronn.ct_reconstruction.geometry.geometry import Geometry as GeometryPyroNN
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d

device = torch.device('cuda')


def setup_geometry(geometry_type, cq500_volume_shape=None, cq500_volume_spacing=None, cq500_volume_origin=None):
    assert geometry_type in ['full', 'small', 'cq500'], 'Select which geometry you need: cq500, full or small.'
    volume_shape = None
    volume_spacing = None
    volume_origin = None
    if geometry_type == 'full':
        volume_shape = reco_shape_full
        volume_spacing = reco_spacing_full
        volume_origin = reco_origin_full
    elif geometry_type == 'small':
        volume_shape = reco_shape_small
        volume_spacing = reco_spacing_small
        volume_origin = reco_origin_small
    elif geometry_type == 'cq500':
        assert (cq500_volume_shape is not None) and (cq500_volume_spacing is not None) and (cq500_volume_origin is not None)
        volume_shape = cq500_volume_shape
        volume_spacing = cq500_volume_spacing
        volume_origin = cq500_volume_origin

    geometry_pyronn = GeometryPyroNN()
    geometry_pyronn.init_from_parameters(volume_shape=volume_shape, volume_spacing=volume_spacing,
                                         detector_shape=detector_shape, detector_spacing=detector_spacing,
                                         number_of_projections=num_projections, angular_range=angular_range,
                                         trajectory=circular_trajectory_2d,
                                         source_isocenter_distance=source_isocenter_distance,
                                         source_detector_distance=source_detector_distance)

    geometry_differentiable = GeometryDifferentiable(volume_shape, volume_origin, volume_spacing, detector_origin,
                                                     detector_spacing)
    angles = torch.from_numpy(np.linspace(0, angular_range, num_projections, endpoint=False)).to(device)
    projection_matrices, _, _ = params_2_proj_matrix(angles, source_detector_distance * torch.ones_like(angles),
                                                     source_isocenter_distance * torch.ones_like(angles),
                                                     torch.zeros_like(angles), torch.zeros_like(angles),
                                                     detector_spacing[0],
                                                     -detector_origin[0] / detector_spacing[0])

    return geometry_pyronn, geometry_differentiable, projection_matrices


def params_2_proj_matrix(angles, dsd, dsi, tx, ty, det_spacing, det_origin):
    ''' compute fan beam projection matrices from parameters for circular trajectory

    :param angles: projection angles in radians
    :param dsd: source to detector distance
    :param dsi: source to isocenter distance
    :param tx: additional detector offset in x (usually 0 for motion free, ideal trajectory)
    :param ty: additional detector offset in y (usually 0 for motion free, ideal trajectory)
    :param det_spacing: detector pixel size
    :param det_origin: attention!! this is (-detector_origin / detector_spacing) or simply (image_size - 0.5)!!
    :return:
    '''
    num_angles = len(angles)
    matrices = torch.zeros((num_angles, 2, 3), device=device)
    matrices[:, 0, 0] = -dsd * torch.sin(angles) / det_spacing + det_origin * torch.cos(angles)
    matrices[:, 0, 1] = dsd * torch.cos(angles) / det_spacing + det_origin * torch.sin(angles)
    matrices[:, 0, 2] = dsd * tx / det_spacing + det_origin * (dsi + ty)
    matrices[:, 1, 0] = torch.cos(angles)
    matrices[:, 1, 1] = torch.sin(angles)
    matrices[:, 1, 2] = dsi + ty

    intrinsics = torch.zeros((num_angles, 2, 2), device=device)
    intrinsics[:, 0, 0] = dsd / det_spacing
    intrinsics[:, 0, 1] = det_origin
    intrinsics[:, 1, 1] = 1.

    extrinsics = torch.zeros((num_angles, 2, 3), device=device)
    extrinsics[:, 0, 0] = - torch.sin(angles)
    extrinsics[:, 0, 1] = torch.cos(angles)
    extrinsics[:, 0, 2] = tx
    extrinsics[:, 1, 0] = torch.cos(angles)
    extrinsics[:, 1, 1] = torch.sin(angles)
    extrinsics[:, 1, 2] = dsi + ty

    assert torch.allclose(matrices, torch.einsum('aij,ajk->aik', intrinsics, extrinsics))

    # normalize w.r.t. lower right entry
    matrices = matrices / matrices[:, 1, 2][:, None, None]

    return matrices, extrinsics, intrinsics


def add_motion(projection_matrices, amplitude_rotation, amplitude_translation):
    with torch.no_grad():
        projection_matrices = torch.moveaxis(projection_matrices, 0, 2)

        # set up the perturbation as required by the motion model
        motion_model = MotionModel2DTorch('spline_akima', num_nodes=10, num_projections=360)
        free_parameters = torch.zeros(motion_model.free_parameters, device=device)
        r = (torch.rand(motion_model.num_nodes) - 0.5) * amplitude_rotation
        tx = (torch.rand(motion_model.num_nodes) - 0.5) * amplitude_translation
        ty = (torch.rand(motion_model.num_nodes) - 0.5) * amplitude_translation
        free_parameters[0::3] = r
        free_parameters[1::3] = tx
        free_parameters[2::3] = ty

        projection_matrices_perturbed = motion_model.eval(free_parameters, projection_matrices,
                                                          return_motion_curves=False, do_zero_center=True,
                                                          is_radian=False)
        projection_matrices_perturbed = torch.moveaxis(projection_matrices_perturbed, 2, 0)

    return projection_matrices_perturbed
