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

import numpy as np
from typing import Tuple
from pyronn.ct_reconstruction.geometry.geometry import Geometry


def ramp(detector_width:int)->np.array:
    filter_array = np.zeros(detector_width)
    frequency_spacing = 0.5 / (detector_width / 2.0)
    for i in range(0, filter_array.shape[0]):
        if i <= filter_array.shape[0] / 2.0:
            filter_array[i] = i * frequency_spacing
        elif i > filter_array.shape[0] / 2.0:
            filter_array[i] = 0.5 - (((i - filter_array.shape[0] / 2.0)) * frequency_spacing)
    return filter_array.astype(np.float32)


def ramp_2D(detector_shape:Tuple[int,int], number_of_projections:int)->np.array:
    detector_width = detector_shape[-1]

    filter = [
        np.reshape(
            ramp(detector_width),
            (1, detector_width)
        )
        for i in range(0, number_of_projections)
    ]

    filter = np.concatenate(filter)

    return filter


def ramp_3D(detector_shape:Tuple[int,int,int], number_of_projections:int)->np.array:
    detector_width = detector_shape[-1]

    filter = [
        np.reshape(
            ramp(detector_width),
            (1, 1, detector_width)
        )
        for i in range(0, number_of_projections)
    ]

    filter = np.concatenate(filter)

    return filter


def ram_lak(detector_width:int, detector_spacing:float)->np.array:
    filter_array = np.zeros(detector_width)
    filter_array[0] = (0.25 / (detector_spacing * detector_spacing))
    odd = (-1.0 / (np.pi * np.pi * detector_spacing * detector_spacing))

    for i in range(1, int(filter_array.shape[0])):
        if i < filter_array.shape[0] / 2:
            if (i % 2) == 1:
                filter_array[i] = odd / (i * i)
        if i >= filter_array.shape[0] / 2:
            tmp = filter_array.shape[0] - i
            if (tmp % 2) == 1:
                filter_array[i] = odd / (tmp * tmp)

    filter_array = np.fft.fft(filter_array)
    return np.real(filter_array).astype(np.float32)    


def ram_lak_2D(detector_shape:Tuple[int,int], detector_spacing:Tuple[float,float], number_of_projections:int)->np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter = [
        np.reshape(
            ram_lak(detector_width, detector_spacing_width),
            (1, detector_width)
        )
        for i in range(0, number_of_projections)
    ]

    filter = np.concatenate(filter)

    return filter


def ram_lak_3D(detector_shape:Tuple[int,int,int], detector_spacing:Tuple[float,float,float], number_of_projections:int)->np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter = [
        np.reshape(
            ram_lak(detector_width, detector_spacing_width),
            (1, 1, detector_width)
        )
        for i in range(0, number_of_projections)
    ]

    filter = np.concatenate(filter)

    return (1 / 1.0) * filter


def shepp_logan(detector_width:int, detector_spacing:float)->np.array:
    filter_array = np.zeros(detector_width)
    odd = (-1.0 / (np.pi * np.pi * detector_spacing * detector_spacing))

    for i in range(0, int(filter_array.shape[0])):
        if i < filter_array.shape[0] / 2:
            filter_array[i] = odd / (4*i * i - 1)
        if i >= filter_array.shape[0] / 2:
            tmp = filter_array.shape[0] - i
            filter_array[i] = odd / (4*tmp * tmp - 1)

    filter_array = np.fft.fft(filter_array)
    return np.real(filter_array).astype(np.float32)    

def shepp_logan_2D(detector_shape:Tuple[int,int], detector_spacing:Tuple[float,float], number_of_projections:int)->np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter = [
        np.reshape(
            shepp_logan(detector_width, detector_spacing_width),
            (1, detector_width)
        )
        for i in range(0, number_of_projections)
    ]

    filter = np.concatenate(filter)

    return filter

def shepp_logan_3D(detector_shape:Tuple[int,int,int], detector_spacing:Tuple[float,float,float], number_of_projections:int)->np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter = [
        np.reshape(
            shepp_logan(detector_width, detector_spacing_width),
            (1, 1, detector_width)
        )
        for i in range(0, number_of_projections)
    ]

    filter = np.concatenate(filter)

    return (1 / 1.0) * filter