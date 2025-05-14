cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(PVDsim LANGUAGES CXX CUDA)

#--- Language standards ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

#--- Dependencies ---
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

#--- Include directories ---
include_directories(${CMAKE_SOURCE_DIR}/include)

#--- Find CUDA device runtime for kernel registration ---
find_library(CUDA_DEVICE_RUNTIME_LIBRARY
    cudadevrt
    HINTS "$ENV{CUDA_PATH}/lib/x64"
)
message(STATUS "CUDA_DEVICE_RUNTIME_LIBRARY: ${CUDA_DEVICE_RUNTIME_LIBRARY}")

#--- Core simulation library ---
add_library(core STATIC
    src/core/chamber.cpp
    src/core/particle.cpp
)
target_include_directories(core PUBLIC include)
target_link_libraries(core PUBLIC CUDA::cudart_static)

#--- GPU + Renderer library ---
# combine update and renderer into one static lib to ensure single device link
add_library(gpu_render STATIC
    src/gpu/update.cu
    src/render/renderer.cu
)
set_target_properties(gpu_render PROPERTIES
    CUDA_SEPARABLE_COMPILATION    ON
    CUDA_RESOLVE_DEVICE_SYMBOLS   ON
)
target_include_directories(gpu_render PUBLIC include)
target_link_libraries(gpu_render PUBLIC
    CUDA::cudart_static
    OpenGL::GL
    glfw
    glad::glad
)

#--- Main executable ---
add_executable(pvd_app
    src/core/main.cpp
)
target_include_directories(pvd_app PRIVATE include)
set_target_properties(pvd_app PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(pvd_app PRIVATE
    core
    gpu_render
    ${CUDA_DEVICE_RUNTIME_LIBRARY}
)
