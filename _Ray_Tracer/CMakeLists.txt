cmake_minimum_required(VERSION 3.18)
project(MyProject LANGUAGES CXX CUDA)

find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Ustawienia builda (Debug, Release)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Dodaj pliki źródłowe
include_directories(inc)
aux_source_directory(src SOURCES)
aux_source_directory(src/CUDA CUDA_SOURCES)

# Znalezienie pakietów
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Domyślne flagi optymalizacji
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -w")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fopenmp -w") # tylko lokalnie
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -march=native -w")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -ftz=true -march=native -w") # tylko lokalnie

# --- Targets dla Linuxa --- #

# # Target dla Linux x86 (32-bit CPU) - float
# set(target_name "cpu_x86_f.exe")
# add_executable(${target_name} ${SOURCES})
# target_compile_definitions(${target_name} PRIVATE LINUX CPU ARCH_X86 UNIT_FLOAT)
# set_target_properties(${target_name} PROPERTIES COMPILE_FLAGS "-m32" LINK_FLAGS "-m32")
# target_link_libraries(${target_name} PUBLIC gomp)

# # Target dla Linux x86 (32-bit CPU) - double
# set(target_name "cpu_x86_d.exe")
# add_executable(${target_name} ${SOURCES})
# target_compile_definitions(${target_name} PRIVATE LINUX CPU ARCH_X86 UNIT_DOUBLE)
# set_target_properties(${target_name} PROPERTIES COMPILE_FLAGS "-m32" LINK_FLAGS "-m32")
# target_link_libraries(${target_name} PUBLIC gomp)

# Target dla Linux x64 (64-bit CPU) - float
set(target_name "cpu_x64_f.exe")
add_executable(${target_name} ${SOURCES})
target_compile_definitions(${target_name} PRIVATE LINUX CPU ARCH_X64 UNIT_FLOAT)
set_target_properties(${target_name} PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64")
target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)

# Target dla Linux x64 (64-bit CPU) - double
set(target_name "cpu_x64_d.exe")
add_executable(${target_name} ${SOURCES})
target_compile_definitions(${target_name} PRIVATE LINUX CPU ARCH_X64 UNIT_DOUBLE)
set_target_properties(${target_name} PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64")
target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)



if (NVIDIA_SMI_EXECUTABLE)
    # Target dla CUDA na Linuxie (64-bit GPU) - float
    set(target_name "gpu_CUDA_f.exe")
    add_executable(${target_name} ${CUDA_SOURCES})
    target_compile_definitions(${target_name} PRIVATE LINUX GPU UNIT_FLOAT)
    target_link_libraries(${target_name} CUDA::cudart)

    # Target dla CUDA na Linuxie (64-bit GPU) - double
    set(target_name "gpu_CUDA_d.exe")
    add_executable(${target_name} ${CUDA_SOURCES})
    target_compile_definitions(${target_name} PRIVATE LINUX GPU UNIT_DOUBLE)
    target_link_libraries(${target_name} CUDA::cudart)
endif()
