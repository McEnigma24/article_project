cmake_minimum_required(VERSION 3.18)
project(MyProject LANGUAGES CXX CUDA)

find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(CAIRO REQUIRED cairo)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GDKPIXBUF REQUIRED gdk-pixbuf-2.0)


# Ścieżka do biblioteki
set(LIBRARY_PATH_MOVIEMAKER_CPP ${CMAKE_CURRENT_SOURCE_DIR}/external/moviemaker-cpp/build/)
set(LIBRARY_PATH_BMP ${CMAKE_CURRENT_SOURCE_DIR}/external/BMP_lib/build/)
set(LIBRARY_PATH_TIMERS ${CMAKE_CURRENT_SOURCE_DIR}/external/Timers_lib/build/)
set(LIBRARY_PATH_RANDOMS ${CMAKE_CURRENT_SOURCE_DIR}/external/Randoms_lib/build/)
set(LIBRARY_PATH_Multithreaded_Memcpy ${CMAKE_CURRENT_SOURCE_DIR}/external/Multithreaded_Memcpy_lib/build/)
set(LIBRARY_PATH_MY_RAY_TRACER_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/sacred_libs)

# Dodajemy ścieżkę do katalogu, w którym jest ta biblioteka
link_directories(${LIBRARY_PATH_MOVIEMAKER_CPP})
link_directories(${LIBRARY_PATH_BMP})
link_directories(${LIBRARY_PATH_TIMERS})
link_directories(${LIBRARY_PATH_RANDOMS})
link_directories(${LIBRARY_PATH_Multithreaded_Memcpy})
link_directories(${LIBRARY_PATH_MY_RAY_TRACER_CUDA})

include_directories(_inc _inc/base external/_Ray_Tracer/inc external/BMP_lib/_inc external/Timers_lib/_inc external/Randoms_lib/_inc external/moviemaker-cpp/include, external/Multithreaded_Memcpy_lib/_inc)
aux_source_directory(_src SOURCES)
aux_source_directory(_src/base SOURCES)

aux_source_directory(_src TEST_SOURCES)
aux_source_directory(_src/base TEST_SOURCES)
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*main\\.c*")
aux_source_directory(_test TEST_SOURCES)

# to see everything
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -g -Wall -Wextra")

# for normal coding
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -g -Wfatal-errors -Werror=uninitialized -Werror=init-self -Werror=reorder")

# for release
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -march=native -w")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fopenmp -O3 -march=native -w")



# TARGET 1 #
set(target_name "cpu_n-buff.exe")
add_executable(${target_name} ${SOURCES})
target_include_directories(${target_name} PRIVATE)
target_compile_definitions(${target_name} PRIVATE LINUX UNIT_DOUBLE UNIT_SIM_FLOAT

CPU
N_BUFF

RENDER_ACTIVE
)
target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX CUDA::cudart

${LIBRARY_PATH_MOVIEMAKER_CPP}/libmoviemaker-cpp.so
${LIBRARY_PATH_BMP}/libBMP.so
${LIBRARY_PATH_TIMERS}/libTimers.so
${LIBRARY_PATH_RANDOMS}/libRandoms.so
${LIBRARY_PATH_Multithreaded_Memcpy}/libMultithreaded_Memcpy.so
${LIBRARY_PATH_MY_RAY_TRACER_CUDA}/libgpu_CUDA_d.exe.a
)

# # TARGET 2 #
# set(target_name "cpu_d-buff-same-obj.exe")
# add_executable(${target_name} ${SOURCES})
# target_include_directories(${target_name} PRIVATE)
# target_compile_definitions(${target_name} PRIVATE LINUX UNIT_DOUBLE UNIT_SIM_FLOAT

# CPU
# D_BUFF
# D_BUFF_SAME_OBJ

# RENDER_ACTIVE
# )
# target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX CUDA::cudart

# ${LIBRARY_PATH_MOVIEMAKER_CPP}/libmoviemaker-cpp.so
# ${LIBRARY_PATH_BMP}/libBMP.so
# ${LIBRARY_PATH_TIMERS}/libTimers.so
# ${LIBRARY_PATH_RANDOMS}/libRandoms.so
# ${LIBRARY_PATH_Multithreaded_Memcpy}/libMultithreaded_Memcpy.so
# ${LIBRARY_PATH_MY_RAY_TRACER_CUDA}/libgpu_CUDA_d.exe.a
# )

# # TARGET 3 #
# set(target_name "cpu_d-buff-diff-obj.exe")
# add_executable(${target_name} ${SOURCES})
# target_include_directories(${target_name} PRIVATE)
# target_compile_definitions(${target_name} PRIVATE LINUX UNIT_DOUBLE UNIT_SIM_FLOAT

# CPU
# D_BUFF
# D_BUFF_DIFF_OBJ

# RENDER_ACTIVE
# )
# target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX CUDA::cudart

# ${LIBRARY_PATH_MOVIEMAKER_CPP}/libmoviemaker-cpp.so
# ${LIBRARY_PATH_BMP}/libBMP.so
# ${LIBRARY_PATH_TIMERS}/libTimers.so
# ${LIBRARY_PATH_RANDOMS}/libRandoms.so
# ${LIBRARY_PATH_Multithreaded_Memcpy}/libMultithreaded_Memcpy.so
# ${LIBRARY_PATH_MY_RAY_TRACER_CUDA}/libgpu_CUDA_d.exe.a
# )

# # TARGET 4 #
# set(target_name "gpu_n-buff.exe")
# add_executable(${target_name} ${SOURCES})
# target_include_directories(${target_name} PRIVATE)
# target_compile_definitions(${target_name} PRIVATE LINUX UNIT_DOUBLE UNIT_SIM_FLOAT

# GPU
# N_BUFF

# RENDER_ACTIVE
# )
# target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX CUDA::cudart

# ${LIBRARY_PATH_MOVIEMAKER_CPP}/libmoviemaker-cpp.so
# ${LIBRARY_PATH_BMP}/libBMP.so
# ${LIBRARY_PATH_TIMERS}/libTimers.so
# ${LIBRARY_PATH_RANDOMS}/libRandoms.so
# ${LIBRARY_PATH_Multithreaded_Memcpy}/libMultithreaded_Memcpy.so
# ${LIBRARY_PATH_MY_RAY_TRACER_CUDA}/libgpu_CUDA_d.exe.a
# )