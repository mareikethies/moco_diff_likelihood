import numpy as np
import torch
from torch import nn
import kornia
import matplotlib.pyplot as plt
import argparse

import pyconrad as pyc
pyc.setup_pyconrad()

from pyronn.ct_reconstruction.layers.projection_3d import ConeProjection3D
from pyronn.ct_reconstruction.layers.backprojection_3d import ConeBackProjection3D
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_3d

def generate_label_projections(label_vol,**kwargs):
    sino = ConeProjection3D().forward(label_vol.cuda(),**kwargs)
    return sino.contiguous()

def iterative_reconstruction():
    # ------------------ Declare Parameters ------------------

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3, help='initial learning rate for adam')
    parser.add_argument('--epoch', dest='num_epochs', type=int, default=1000, help='# of epoch')
    args = parser.parse_args()

    # vol_shape= [256,256,256]
    vol_shape= [512,512,512]
    vol_spacing = [1.,1.,1.]

    # detector_width = 370
    # detector_height = 370
    detector_width = 725
    detector_height = 725
    det_shape = [detector_height,detector_width]
    det_spacing=[1.,1.]
    number_of_projections = 180
    angular_range = 2*np.pi
    
    geometry = Geometry()
    geometry.init_from_parameters(volume_shape=vol_shape,volume_spacing=vol_spacing,
                                 detector_shape=det_shape,detector_spacing=det_spacing,
                                 number_of_projections=number_of_projections,angular_range=angular_range,
                                 trajectory=circular_trajectory_3d, source_isocenter_distance=950, source_detector_distance=1200)
    geometry.parameter_dict['volume_origin'] = -(geometry.parameter_dict['volume_shape'] - 1) / 2.0 * geometry.parameter_dict['volume_spacing']


    vol = shepp_logan.shepp_logan_3d(np.asarray(vol_shape))
    sinogram = generate_label_projections(torch.tensor(np.expand_dims(vol,0), dtype=torch.float32,device='cuda').contiguous(), **geometry)
    sinogram_shape = [number_of_projections, detector_height, detector_width]


    zero_vector = np.zeros(np.shape(vol), dtype=np.float32)
    iter_pipeline = pipeline(args, geometry)
    iter_pipeline.train(zero_vector,sinogram)
    
    print("training result:")
    pyc.imshow(np.squeeze(iter_pipeline.result), "Result")
    a=5

class pipeline(object):

    def __init__(self, args, geometry):
        self.args = args
        self.geometry = geometry
        self.model = iterative_module(geometry.volume_shape)
        self.regularizer_weight = 0.0001
        self.lossMSE = nn.MSELoss()
        self.lossTV = kornia.losses.TotalVariation()
        self.optimizer = torch.optim.Adam(self.model.parameters())


    def loss(self, prediction, label,current_reco,  regularizer = False):
        mse = self.lossMSE(prediction, label)
        tv_loss = 0
        if regularizer:
            tv_loss = self.lossTV(current_reco)
        return torch.sum(torch.sum( mse ) + self.regularizer_weight * tv_loss)



    def train_step(self, input, label):

        predictions, current_reco = self.model.forward(**self.geometry)
        self.current_loss = self.loss(predictions, label, current_reco, False)
        self.current_loss.backward()
        self.optimizer.step()



    def train(self, zero_vector, label_data):
        

        last_loss = 100000000
        self.model.train()
        for epoch in range(self.args.num_epochs):
                        
            self.train_step(zero_vector, label_data)

            if epoch % 25 == 0:
                pyc.imshow(self.model.reco.cpu().detach().numpy(), 'reco')
            if epoch % 100 == 0:
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch, self.current_loss.cpu().detach().numpy()))
            if self.current_loss.cpu().detach().numpy() > last_loss*1.03:
                print('break at epoch', epoch)
                break
            last_loss = self.current_loss.cpu().detach().numpy()

        print('training finished')
        self.result = self.model.reco.cpu().detach().numpy()

class iterative_module(nn.Module):
    def __init__(self, shape):
        super(iterative_module, self).__init__()
        self.reco = nn.Parameter(torch.unsqueeze(torch.zeros(shape,dtype=torch.float32).cuda().contiguous(),0))

        self.projection = ConeProjection3D(hardware_interp=False)
    def forward(self, **kwargs):
        # x = torch.add(x,self.reco)
        x = self.projection.forward(self.reco,**kwargs)
        return x,self.reco


if __name__ == '__main__':
    iterative_reconstruction()