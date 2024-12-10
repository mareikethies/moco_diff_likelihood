# Copyright [2019] [Christopher Syben, Markus Michen]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from io import FileIO
import numpy as np
import torch
from typing import Callable, Tuple
import warnings

class Geometry:
    """
        The Base Class for the different Geometries. Provides commonly used members.
    """

    def __init__(self):
        """
            Constructor of Geometry Class.
        Args:
            volume_shape:               The volume size in Z, Y, X order.
            volume_spacing:             The spacing between voxels in Z, Y, X order.
            detector_shape:             Shape of the detector in Y, X order.
            detector_spacing:           The spacing between detector voxels in Y, X order.
            number_of_projections:      Number of equidistant projections.
            angular_range:              The covered angular range.
            trajectory->array:          The function pointer which computes the trajectory.
            source_detector_distance:   The source to detector distance (sdd). Only for fan and cone-beam geometries. Default = 0.
            source_isocenter_distance:  The source to isocenter distance (sid). Only for fan and cone-beam geometries. Default = 0.
            trajectory:Callable         Function pointer to the trajectory compute function.
            projection_multiplier:      Constant factor of the distance weighting for FDK Cone-beam reconstruction. Is computed automatically from geometry.
            step_size:                  Step size for sampling points along the ray in the Cone-beam projector. Default=0.2* voxel_spacing
            swap_detector_axis:         2D Detector axis direction. Default is [1,0,0], swaped [-1,0,0]

        """
        self.gpu_device = True
        self.np_dtype = np.float32  # datatype for np.arrays make sure everything will be float32
        self.parameter_dict = {}
        # Needed parameters
        self.parameter_dict.setdefault('volume_shape',None)
        self.parameter_dict.setdefault('volume_spacing',None)
        self.parameter_dict.setdefault('detector_shape',None)
        self.parameter_dict.setdefault('detector_spacing',None)
        self.parameter_dict.setdefault('number_of_projections',None)
        self.parameter_dict.setdefault('angular_range',None)
        self.parameter_dict.setdefault('trajectory',None)
        # Optional parameters, neccessary for fan- and cone-beam geometry
        self.parameter_dict.setdefault('source_detector_distance',None)
        self.parameter_dict.setdefault('source_isocenter_distance',None)
        # Optional paramters, neccessarry for cone-beam geometry
        self.parameter_dict.setdefault('projection_multiplier',None)
        self.parameter_dict.setdefault('step_size',None)
        self.parameter_dict.setdefault('swap_detector_axis', False)




    def init_from_parameters(self, volume_shape:Tuple[int,...], volume_spacing:Tuple[float,...],
                                    detector_shape:Tuple[int,...], detector_spacing:Tuple[float,...],
                                    number_of_projections:int, angular_range:float,trajectory:Callable,
                                    source_detector_distance:float=.0, source_isocenter_distance:float=.0, swap_detector_axis:bool=False)->None:
        self.parameter_dict['swap_detector_axis'] = swap_detector_axis
        # Volume Parameters:
        self.parameter_dict['volume_shape'] = np.array(volume_shape)
        self.parameter_dict['volume_spacing'] = np.array(volume_spacing, dtype=self.np_dtype)
        self.parameter_dict['volume_origin'] = -(self.parameter_dict['volume_shape'] - 1) / 2.0 * self.parameter_dict['volume_spacing']

        # Detector Parameters:
        self.parameter_dict['detector_shape'] = np.array(detector_shape)
        self.parameter_dict['detector_spacing'] = np.array(detector_spacing, dtype=self.np_dtype)
        self.parameter_dict['detector_origin'] = -(self.parameter_dict['detector_shape'] - 1) / 2.0 * self.parameter_dict['detector_spacing']

        # Trajectory Parameters:
        self.parameter_dict['number_of_projections'] = number_of_projections
        self.parameter_dict['angular_range'] = angular_range    
        self.parameter_dict['sinogram_shape'] = np.array([ self.parameter_dict['number_of_projections'], *self.parameter_dict['detector_shape']])
        self.parameter_dict['source_detector_distance'] = source_detector_distance
        self.parameter_dict['source_isocenter_distance'] = source_isocenter_distance
        self.parameter_dict['trajectory'] = trajectory(**self.parameter_dict)

        # Containing the constant part of the distance weight and discretization invariant
        self.parameter_dict['projection_multiplier'] = self.parameter_dict['source_isocenter_distance'] * self.parameter_dict['source_detector_distance'] * detector_spacing[-1] * np.pi /  self.parameter_dict['number_of_projections']
        self.parameter_dict['step_size'] = 0.2
        

    def to_json(self, path:str)->None:
        """
            Saves the geometry as json in the file denotes by @path
            The internal parameter_dict is stored including the trajectory array, hence, changes in the json file are not validated if correct or not.
            TODO: Future feature should be to store only the core parameters and recompute the origins, trajectory, etc. from core parameters when loaded via from_json() method.
        """
        import json
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        try:
            with open(path, 'w+') as outfile:
                json.dump(self.parameter_dict, outfile,cls=NumpyEncoder)
        except FileNotFoundError and FileExistsError:
            print('Error while saving geometry to file.')
        
    @staticmethod 
    def from_json(path: str):
        import json        
        loaded_geom = Geometry()
        try:
            with open(path,'r') as infile:
                loaded_geom.parameter_dict = json.load(infile)
        except FileNotFoundError:
            print('Error while loading geometry from file.')
        for key, value in loaded_geom.parameter_dict.items():
            if isinstance(value,list):
                loaded_geom.parameter_dict[key] = np.asarray(value)
            
        return loaded_geom

    def cuda(self)->None:
        self.gpu_device = True
    
    def cpu(self)->None:
        self.gpu_device = False

    def keys(self)->str:
        return self.parameter_dict

    def __getitem__(self,key:str)->torch.Tensor:
        try:
            parameter = self.parameter_dict[key]
            if hasattr(parameter,'__len__'):                               
                tmp_tensor = torch.Tensor(parameter)
            else:
                tmp_tensor = torch.Tensor([parameter])
        except:
            print('Attribute <'+key+'> could not be transformed to torch.Tensor')            
        if self.gpu_device:
            return tmp_tensor.cuda()
        else:
            return tmp_tensor.cpu()

    def __generate_trajectory__(self)->None:
        # TODO: For now cone-beam circular full scan is fixed as initialization. Need to be reworked such that header defines the traj function
        from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_3d
        self.parameter_dict['trajectory'] = circular_trajectory_3d(**self.parameter_dict)

    def fan_angle(self)->float:
        return np.arctan(((self.parameter_dict['detector_shape'][-1] - 1) / 2.0 * self.parameter_dict['detector_spacing'][-1]) / self.parameter_dict['source_detector_distance'])

    def cone_angle(self)->float:
        return np.arctan(((self.parameter_dict['detector_shape'][-2] - 1) / 2.0 * self.parameter_dict['detector_spacing'][-2]) / self.parameter_dict['source_detector_distance'])

    def set_detector_shift(self,detector_shift:Tuple[float,...])->None:
        """
            Applies a detector shift in px to the geometry.
            This triggers a recomputation of the trajectory. Projection matrices will be overwritten.
            
            :param detector_shift: Tuple[float,...] with [y,x] convention in Pixels
        """
        #change the origin according to the shift
        self.parameter_dict['detector_origin'] = self.parameter_dict['detector_origin'] + (detector_shift*self.parameter_dict['detector_spacing'])
        #recompute the trajectory with the applied shift
        # TODO: For now cone-beam circular full scan is fixed as initialization. Need to be reworked such that header defines the traj function
        self.__generate_trajectory__()
    
    def swap_axis(self, swap_det_axis:bool)->None:
        """
            Sets the direction of the rotatation of the system. 
            This triggers a recomputation of the trajectory. Projection matrices will be overwritten.

            :param counter_clockwise: wether the system rotates counter clockwise (True) or not.
        """
        self.parameter_dict['swap_det_axis'] = swap_det_axis
        self.__generate_trajectory__()

    
    @property
    def volume_shape(self) -> Tuple[int,...]:
        return tuple(self.parameter_dict['volume_shape'])

    @property
    def volume_spacing(self) -> Tuple[float,...]:
        return tuple(self.parameter_dict['volume_spacing'])

    @property
    def detector_shape(self) -> Tuple[int,...]:
        return tuple(self.parameter_dict['detector_shape'])

    @property
    def detector_spacing(self) -> Tuple[float,...]:
        return tuple(self.parameter_dict['detector_spacing'])

    @property
    def number_of_projections(self) -> int:
        return int(self.parameter_dict['number_of_projections'])

    @property
    def angular_range(self) -> float:
        return self.parameter_dict['angular_range']

    @property
    def trajectory(self) -> Tuple[float,...]:
        return self.parameter_dict['trajectory']

    @property
    def source_detector_distance(self) -> float:
        return self.parameter_dict['source_detector_distance']
    
    @property
    def source_isocenter_distance(self) -> float:
        return self.parameter_dict['source_isocenter_distance']

    @property
    def projection_multiplier(self) -> float:
        return self.parameter_dict['projection_multiplier']
    
    @property
    def step_size(self) -> float:
        return self.parameter_dict['step_size']
    
    @property
    def swap_detector_axis(self) -> bool:
        return self.parameter_dict['swap_detector_axis']

    @property
    def is_gpu(self) -> bool:
        return self.gpu_device