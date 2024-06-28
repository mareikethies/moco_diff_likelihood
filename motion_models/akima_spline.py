import torch


def interpolate_akima_spline_alternative(node_positions, node_values, interpolation_points):
    ''' Pytorch implementation of Akima splines following formulas in Wikipedia

    :param node_positions: x values of nodes
    :param node_values: y values of nodes
    :param interpolation_points: x values of interpolation points
    :return: y values of interpolation points
    '''
    num_nodes = node_positions.shape[0]

    dx = torch.diff(node_positions)
    dy = torch.diff(node_values)

    m = dy / dx

    dm = torch.abs(torch.diff(m))

    s = torch.zeros_like(node_values)
    # first two and last two need special formula
    s[0] = m[0]
    s[1] = (m[0] + m[1]) / 2.
    s[num_nodes - 1] = m[num_nodes - 2]
    s[num_nodes - 2] = (m[num_nodes - 2] + m[num_nodes - 3]) / 2.
    # fill the rest
    s[2:num_nodes - 2] = dm[2:num_nodes - 2] * m[1:num_nodes - 3] + dm[0:num_nodes - 4] * m[2:num_nodes - 2] / (dm[2:num_nodes - 2] + dm[0:num_nodes - 4])

    a = node_values[0:-1]
    b = s[0:-1]
    c = (3 * m - 2 * s[0:-1] - s[1:]) / (node_positions[1:] - node_positions[0:-1])
    d = (s[0:-1] + s[1:] - 2 * m) / torch.square(node_positions[1:] - node_positions[0:-1])

    # find out which piecewise spline needs to be applied for which interpolation point
    spline_piece = torch.zeros_like(interpolation_points)
    for i in range(num_nodes - 1):
        spline_piece[torch.where(interpolation_points >= node_positions[i])] = i

    # do the interpolation
    xi = node_positions[spline_piece]
    out = a[spline_piece] + b[spline_piece] * (interpolation_points - xi) + c[spline_piece] * torch.pow((interpolation_points - xi), 2) + d[spline_piece] * torch.pow((interpolation_points - xi), 3)

    return out


def interpolate_akima_spline(node_positions, node_values, interpolation_points):
    ''' Pytorch implementation of Akima splines following implementations in Matlab and scipy

    :param node_positions: x values of nodes
    :param node_values: y values of nodes
    :param interpolation_points: x values of interpolation points
    :return: y values of interpolation points
    '''
    assert node_positions.shape == node_values.shape, 'Node positions and values must be of same length.'
    # assert torch.all(interpolation_points >= node_positions[0]), \
    #     'All interpolation points should be greater or equal than the smallest node position.'
    # assert torch.all(interpolation_points <= node_positions[-1]), \
    #     'All interpolation points should be smaller or equal than the largest node position.'

    num_nodes = node_positions.shape[0]

    dx = torch.diff(node_positions)
    dy = torch.diff(node_values)

    assert torch.all(dx > 0), 'Node positions must be in ascending order.'

    m = dy / dx

    mm = torch.unsqueeze(2 * m[0] - m[1], 0)
    mmm = 2 * mm - m[0]
    mp = torch.unsqueeze(2 * m[num_nodes - 2] - m[num_nodes - 3], 0)
    mpp = 2 * mp - m[num_nodes - 2]
    m1 = torch.cat((mmm, mm, m, mp, mpp))

    dm = torch.abs(torch.diff(m1))
    f1 = dm[2:num_nodes + 2]
    f2 = dm[0:num_nodes]
    f12 = f1 + f2
    id = torch.where(f12 > 1e-8 * torch.max(f12))[0]
    b = m1[1:num_nodes + 1]
    b[id] = (f1[id] * m1[id + 1] + f2[id] * m1[id + 2]) / f12[id]
    c = (3 * m - 2 * b[0:num_nodes - 1] - b[1:num_nodes]) / dx
    d = (b[0:num_nodes - 1] + b[1:num_nodes] - 2 * m) / torch.square(dx)

    # find out which piecewise spline needs to be applied for which interpolation point
    spline_piece = torch.zeros_like(interpolation_points)
    for i in range(num_nodes - 1):
        spline_piece[torch.where(interpolation_points >= node_positions[i])] = i

    # do the interpolation
    xi = node_positions[spline_piece]
    out = node_values[spline_piece] + b[spline_piece] * (interpolation_points - xi) + c[spline_piece] * torch.pow((interpolation_points - xi), 2) + d[spline_piece] * torch.pow((interpolation_points - xi), 3)

    return out
