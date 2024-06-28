import torch
from akima_spline import interpolate_akima_spline
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


class MotionModel2DTorch:
    def __init__(self, selection, **kwargs):
        ''' Creates a 2D fan-beam motion model.

        :param selection: string selecting one of the types below
        :param kwargs: selection specific additional arguments like number of projections/ number of spline nodes
        '''
        if selection == 'rigid_2d':
            assert 'num_projections' in kwargs.keys(), 'Please provide the num_projections argument for the motion model.'
            self.free_parameters_per_node = 3
            self.free_parameters = self.free_parameters_per_node * kwargs['num_projections']
            self.eval = self.rigid_2d
        elif selection == 'spline_cubic':
            assert 'num_nodes' in kwargs.keys(), 'Please provide the num_nodes argument for the motion model.'
            assert 'num_projections' in kwargs.keys(), 'Please provide the num_projections argument for the motion model.'
            self.free_parameters_per_node = 3
            self.free_parameters = self.free_parameters_per_node * kwargs['num_nodes']
            self.num_nodes = kwargs['num_nodes']
            self.eval = self.spline_cubic
            # nodes: a 1D numpy array of increasing values indicating at which projection indices the spline nodes are,
            # if None, then evenly spaced along the projections with both endpoints included
            if 'nodes' in kwargs.keys() and (kwargs['nodes'] is not None):
                self.nodes = kwargs['nodes']
            else:
                self.nodes = torch.linspace(0, kwargs['num_projections'] - 1, steps=self.num_nodes)
        elif selection == 'stepwise_rigid':
            assert 'num_projections' in kwargs.keys(), 'Please provide the num_projections argument for the motion model.'
            assert 'start_projection' in kwargs.keys(), 'Please provide the projection index where the step starts.'
            assert 'step_length' in kwargs.keys(), 'Please provide the number of projection which the step lasts.'
            assert kwargs['start_projection'] + kwargs['step_length'] < kwargs['num_projections']
            self.free_parameters = 3
            self.start_projection = kwargs['start_projection']
            self.step_length = kwargs['step_length']
            self.eval = self.stepwise_rigid
        elif selection == 'spline_akima':
            assert 'num_nodes' in kwargs.keys(), 'Please provide the num_nodes argument for the motion model.'
            assert 'num_projections' in kwargs.keys(), 'Please provide the num_projections argument for the motion model.'
            self.num_projections = kwargs['num_projections']
            self.free_parameters_per_node = 3
            self.free_parameters = self.free_parameters_per_node * kwargs['num_nodes']
            self.num_nodes = kwargs['num_nodes']
            self.eval = self.spline_akima
            # nodes: a 1D numpy array of increasing values indicating at which projection indices the spline nodes are,
            # if None, then evenly spaced along the projections with both endpoints included
            if 'nodes' in kwargs.keys() and (kwargs['nodes'] is not None):
                assert len(kwargs['nodes']) == self.num_nodes, 'Make sure that the length of the nodes list is consistent with num_nodes.'
                self.nodes = torch.FloatTensor(kwargs['nodes'])
            else:
                self.nodes = torch.linspace(0, kwargs['num_projections'] - 1, steps=self.num_nodes)
        else:
            print('This model is not implemented.')
            raise

    def rigid_2d(self, free_params, projection_matrices_input, do_zero_center=False, is_radian=True):
        '''Computes out = P @ M for M being a 2d rigid transformation matrix

        :param free_params: params for M; (r, tx, ty) for each projection as 1D torch tensor of size 3*num_projections
        :param projection_matrices_input: the best guess for good projection matrices as 3D torch tensor of size
        2x3xnum_projections
        :return: the motion adjusted projection matrices as 3D torch tensor of size 2x3xnum_projections
        '''
        num_projections = projection_matrices_input.shape[2]
        assert (free_params.shape[0] == self.free_parameters), 'Wrong input to motion model rigid_2d.'

        if do_zero_center:
            for i in range(3):
                free_params[i::3] = free_params[i::3] - torch.mean(free_params[i::3])

        # convert to radian if rotation are given in degree
        if not is_radian:
            free_params[0::3] = free_params[0::3] / 180. * torch.pi

        rotations = torch.zeros((2, 2, num_projections), device=free_params.get_device())
        rotations[0, 0, :] = torch.cos(free_params[0::3])
        rotations[0, 1, :] = -torch.sin(free_params[0::3])
        rotations[1, 0, :] = torch.sin(free_params[0::3])
        rotations[1, 1, :] = torch.cos(free_params[0::3])

        translations = torch.zeros((2, 1, num_projections), device=free_params.get_device())
        translations[0, :, :] = free_params[1::3]
        translations[1, :, :] = free_params[2::3]

        # lower row of 0s and 1s to make a 4x4 transformation matrix
        lower_row = torch.zeros((1, 3, num_projections), device=free_params.get_device())
        lower_row[:, 2, :] = 1

        rigid_transform = torch.cat((torch.cat((rotations, translations), 1), lower_row), 0)
        # apply matrix multiplication along third dimension
        out = torch.einsum('ijn,jkn->ikn', projection_matrices_input, rigid_transform)

        return out

    def spline_cubic(self, free_params, projection_matrices_input, return_motion_curves=False):
        ''' Models a 2d rigid motion trajectory as 3 individual akima splines for r, tx, ty

        :param free_params: params for spline nodes; (r, tx, ty) for each node as 1D torch tensor of size 3*num_nodes
        :param projection_matrices_input: the best guess for good projection matrices as 3D torch tensor of size
        2x3xnum_projections
        :return: the motion adjusted projection matrices as 3D torch tensor of size 2x3xnum_projections
        '''
        num_projections = projection_matrices_input.shape[2]
        assert (free_params.shape[0] == self.free_parameters), 'Wrong input to motion model spline_akima.'
        self.nodes = self.nodes.to(free_params.get_device())

        r = torch.unsqueeze(free_params[0::3], 1)
        tx = torch.unsqueeze(free_params[1::3], 1)
        ty = torch.unsqueeze(free_params[2::3], 1)
        motion_types = [r, tx, ty]
        # do the interpolation, one spline per parameter
        interpolated_values = torch.zeros((num_projections, 3), device=free_params.get_device())

        for i in range(3):
            coeffs = natural_cubic_spline_coeffs(self.nodes, motion_types[i])
            spline = NaturalCubicSpline(coeffs)
            evaluation_points = torch.arange(num_projections, device=free_params.get_device())
            interpolated_values[:, i] = torch.squeeze(spline.evaluate(evaluation_points))

        out = torch.zeros(num_projections * 3, device=free_params.get_device())
        out[0::3] = interpolated_values[:, 0]
        out[1::3] = interpolated_values[:, 1]
        out[2::3] = interpolated_values[:, 2]

        motion_model_rigid = MotionModel2DTorch('rigid_2d', num_projections=num_projections)
        if return_motion_curves:
            return motion_model_rigid.eval(out, projection_matrices_input), (out[0::3], out[1::3], out[2::3])
        else:
            return motion_model_rigid.eval(out, projection_matrices_input)

    def stepwise_rigid(self, free_params, projection_matrices_input, return_motion_curves=False):
        num_projections = projection_matrices_input.shape[2]
        assert (free_params.shape[0] == self.free_parameters), 'Wrong input to motion model stepwise_rigid.'
        interpolated_values = torch.zeros((num_projections, 3), device=free_params.get_device())
        interpolated_values[self.start_projection + self.step_length:, :] = free_params
        if (self.start_projection + self.step_length) < num_projections:
            interpolated_values[self.start_projection:self.start_projection + self.step_length, 0] = torch.linspace(0, float(free_params[0]), steps=self.step_length)
            interpolated_values[self.start_projection:self.start_projection + self.step_length, 1] = torch.linspace(0, float(free_params[1]), steps=self.step_length)
            interpolated_values[self.start_projection:self.start_projection + self.step_length, 2] = torch.linspace(0, float(free_params[2]), steps=self.step_length)

        out = torch.zeros(num_projections * 3, device=free_params.get_device())
        out[0::3] = interpolated_values[:, 0]
        out[1::3] = interpolated_values[:, 1]
        out[2::3] = interpolated_values[:, 2]

        motion_model_rigid = MotionModel2DTorch('rigid_2d', num_projections=num_projections)
        if return_motion_curves:
            return motion_model_rigid.eval(out, projection_matrices_input), (out[0::3], out[1::3], out[2::3])
        else:
            return motion_model_rigid.eval(out, projection_matrices_input)

    def spline_akima(self, free_params, projection_matrices_input, return_motion_curves=False, do_zero_center=False,
                     is_radian=True):
        ''' Models a 3d rigid motion trajectory as 6 individual akima splines for rx, ry, rz, tx, ty, tz

        :param free_params: params for spline nodes; (r, tx, ty) for each node as 1D torch tensor of size 3*num_nodes
        :param projection_matrices_input: the best guess for good projection matrices as 3D torch tensor of size
        2x3xnum_projections
        :param return_motion_curves: whether the interpolated motion curves are returned from the function
        :param do_zero_center: whether the mean is subtracted from each motion curve
        :param is_radian: whether the rotational components are assumed to be radian, else degree

        :return: the motion adjusted projection matrices as 3D torch tensor of size 3x4xnum_projections
        '''
        num_projections = projection_matrices_input.shape[2]

        assert (free_params.shape[0] == self.free_parameters), 'Wrong input to motion model spline_akima.'
        self.nodes = self.nodes.to(free_params.device)

        r = free_params[0::3]
        tx = free_params[1::3]
        ty = free_params[2::3]

        # convert to radian if rotation are given in degree
        if not is_radian:
            r = r / 180. * torch.pi

        motion_types = [r, tx, ty]
        # do the interpolation, one spline per parameter
        interpolated_values = torch.zeros((num_projections, 3), device=free_params.device)

        means = []
        evaluation_points = torch.arange(num_projections, device=free_params.device)
        for i in range(3):
            interpolated_values[:, i] = interpolate_akima_spline(self.nodes, motion_types[i], evaluation_points)
            means.append(torch.mean(interpolated_values[:, i]))

        out = torch.zeros(num_projections * 3, device=free_params.device)
        out[0::3] = interpolated_values[:, 0]
        out[1::3] = interpolated_values[:, 1]
        out[2::3] = interpolated_values[:, 2]

        motion_model_rigid = MotionModel2DTorch('rigid_2d', num_projections=num_projections)
        perturbed_matrices = motion_model_rigid.eval(out, projection_matrices_input, do_zero_center=do_zero_center)

        # convert back to degrees for returning motion curves
        out = out.detach().clone()
        if not is_radian:
            out[0::3] = out[0::3] * 180. / torch.pi
            means[0] = means[0] * 180. / torch.pi

        if return_motion_curves:
            if do_zero_center:
                return perturbed_matrices, \
                    ((out[0::3], out[1::3], out[2::3]),
                     (self.nodes, self.nodes, self.nodes),
                     (free_params[0::3] - means[0], free_params[1::3] - means[1], free_params[2::3] - means[2]))
            else:
                return perturbed_matrices, \
                    ((out[0::3], out[1::3], out[2::3]),
                     (self.nodes, self.nodes, self.nodes),
                     (free_params[0::3], free_params[1::3], free_params[2::3]))
        else:
            return perturbed_matrices
