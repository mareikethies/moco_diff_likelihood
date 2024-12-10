import numpy as np
from numpy.core.function_base import geomspace
import torch
from torch import nn
import kornia
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.layers.reconstruction import FanBeamReconstruction
from pyronn.ct_reconstruction.layers.projection_2d import FanProjection2D
from pyronn.ct_reconstruction.helpers.filters.filters import shepp_logan_2D
from pyronn.ct_reconstruction.helpers.phantoms.primitives_2d import circle,rect
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d

def def_geometry()->Geometry:
    volume_shape = [256,256]
    volume_spacing = [1,1]
    number_of_projections = 180
    detector_width = 365
    detector_shape = [detector_width]
    detector_spacing = [1]
    angular_range = np.pi

    sdd = 1200
    sid = 950
    geometry = Geometry()
    geometry.init_from_parameters(volume_shape=volume_shape,volume_spacing=volume_spacing,
                                detector_shape=detector_shape,detector_spacing=detector_spacing,
                                number_of_projections=number_of_projections,angular_range=angular_range,
                                trajectory=circular_trajectory_2d, source_isocenter_distance=sid, source_detector_distance=sdd)
    return geometry

    

def calc_trainigdata(n,geometry:Geometry):
    inputs = []
    labels = []
    for i in range(0,n):
        type = np.random.randint(0,2)
        tmp = None
        if type == 0:    
            position = np.random.randint(geometry.volume_shape[-1]*0.05,geometry.volume_shape[-1]*0.75,2)   
            radius = np.random.randint(np.min(position)*0.45,np.min(position),1)                 
            tmp = circle(geometry.volume_shape,position,radius) 
        if type == 1:
            position = np.random.randint(geometry.volume_shape[-1]*0.05,geometry.volume_shape[-1]*0.75,2) 
            length = np.random.randint(position*0.45,position,2)     
            tmp = rect(geometry.volume_shape,position,length)
        sino = FanProjection2D().forward(torch.unsqueeze(torch.tensor(tmp),0).cuda(),**geometry).cpu().numpy()
        inputs.append(sino)
        labels.append(tmp)

    return inputs,labels

geometry = def_geometry()
inputs,labels = calc_trainigdata(100,geometry)
filter_array = shepp_logan_2D(geometry.detector_shape, geometry.detector_spacing, geometry.number_of_projections)
weighting_array = np.zeros_like(filter_array)
model = FanBeamReconstruction(geometry,filter_array,short_scan=True,redundancy_weights=weighting_array,trainable_filter=True, trainable_redundancy_weights=True)

lossf = kornia.losses.TotalVariation()
optimizer = torch.optim.Adam(model.parameters())
# geometry.parameter_dict['volume_shape'] = [512,1024,1024]
for epoch in range(10):
    model.train()
    pred = model.forward(torch.Tensor(inputs).contiguous(), **geometry)
    # pred = model.forward(torch.Tensor(input_data.cpu().numpy()).cuda().contiguous(), **geometry)

    loss = lossf.forward(pred.cpu())
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()