from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='extend_cpp',
    ext_modules=[
        CppExtension(
            name='faceConnected',
            sources=['src/connected.cpp'],
            extra_compile_args={
                "cxx": ["-O2", "-fopenmp", "-std=c++17"]
            },
            extra_link_args=["-fopenmp"],  
        ),
        CppExtension(
            name='svo',
            sources=['src/bindings.cpp', 'src/octree.cpp', 'src/utils.cpp'],
            include_dirs=["./include"],
            extra_compile_args={
                "cxx": ["-O2", "-I./include", "-fopenmp", "-std=c++17"]
            },
            extra_link_args=["-fopenmp"],
        ),
        CUDAExtension(
            name='planar',
            sources=['src/planar.cpp', 'src/planar_gpu.cu'],
            include_dirs=["./include"],
            extra_compile_args={
                "cxx": ["-O2", "-I./include", "-fopenmp", "-std=c++17"],
                "nvcc": ["-O2", "-I./include"],
            },
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
