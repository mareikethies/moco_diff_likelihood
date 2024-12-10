from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
name='pyronn_layers',
ext_modules=[
    CUDAExtension('pyronn_layers', [
        #Python Bindings
        'pyro-nn-layers/src/torch_ops/pyronn_torch_layers.cc',
        #Parallel operators
        'pyro-nn-layers/src/torch_ops/par_projector_2D_OPKernel.cc', 
        'pyro-nn-layers/src/kernels/par_projector_2D_CudaKernel.cu',
        'pyro-nn-layers/src/torch_ops/par_backprojector_2D_OPKernel.cc', 
        'pyro-nn-layers/src/kernels/par_backprojector_2D_CudaKernel.cu',
        #Fan operators
        'pyro-nn-layers/src/torch_ops/fan_projector_2D_OPKernel.cc', 
        'pyro-nn-layers/src/kernels/fan_projector_2D_CudaKernel.cu',
        'pyro-nn-layers/src/torch_ops/fan_backprojector_2D_OPKernel.cc', 
        'pyro-nn-layers/src/kernels/fan_backprojector_2D_CudaKernel.cu',
        # #Cone operators
        'pyro-nn-layers/src/torch_ops/cone_projector_3D_OPKernel.cc', 
        'pyro-nn-layers/src/kernels/cone_projector_3D_CudaKernel.cu',
        'pyro-nn-layers/src/kernels/cone_projector_3D_CudaKernel_hardware_interp.cu',
        'pyro-nn-layers/src/torch_ops/cone_backprojector_3D_OPKernel.cc', 
        'pyro-nn-layers/src/kernels/cone_backprojector_3D_CudaKernel.cu',
        'pyro-nn-layers/src/kernels/cone_backprojector_3D_CudaKernel_hardware_interp.cu',
    ],
    # extra_compile_args=['-g']
    ),
],
cmdclass={
    'build_ext': BuildExtension
})
