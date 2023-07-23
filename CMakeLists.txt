cmake_minimum_required(VERSION 3.25)
project(KmeansCUDA LANGUAGES CUDA CXX)

find_package(CUDAToolkit)
set(CMAKE_CUDA_STANDARD 17)

add_executable(KmeansCUDA
        main.cu
        KmeansInitializer.cpp
        KmeansInitializer.cpp
        Point.h
        Kluster.h
        Kluster.cpp
        sequential/KmeansSequentialSolver.h
        sequential/KmeansSequentialSolver.cpp
        ParallelCUDA/KmeansParallelCUDASolver.h
        ParallelCUDA/KmeansParallelCUDASolver.cpp
        ParallelCUDA/KmeansCUDA.cuh
        ParallelCUDA/KmeansCUDA.cu
        )

target_link_libraries(KmeansCUDA PRIVATE CUDA::cudart)
set_target_properties(KmeansCUDA PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        )

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_86,code=sm_86")