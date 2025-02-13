#include "__preprocessor__.h"
#include "__time_stamp__.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "movie.h"

#include "RT_Renderer.h"
#include "CTRL_Setuper.h"

#define CCE(x) { cudaError_t err = x;  if (err != cudaSuccess) { const string error = "CUDA ERROR - " + std::to_string(__LINE__) + " : " + __FILE__ + "\n"; cout << error; exit(EXIT_FAILURE);} }

__global__ void test(int* a, int* b, int* result, int ARRAY_SIZE)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(!(i < ARRAY_SIZE)) return;

    result[i] = a[i] + b[i];
}

void OpenMP_GPU_test()
{
    int size = 1000000;
    int* a = new int[size];
    int* b = new int[size];
    int* result = new int[size];

    for(int i=0; i<size; i++)
    {
        a[i] = i;
        b[i] = size - i;
    }

    time_stamp_reset();

    for(int i=0; i<size; i++)
    {
        result[i] = a[i] + b[i];
    }
    time_stamp("Iterative");

    #pragma omp parallel for schedule(static)
    for(int i=0; i<size; i++)
    {
        result[i] = a[i] + b[i];
    }
    time_stamp("Parallel");
        
    int byte_size = size * sizeof(int);
    int* dev_a{};
    int* dev_b{};
    int* dev_result{};
    CCE(cudaSetDevice(0));
    CCE(cudaMalloc((void**)&dev_a, byte_size));
    CCE(cudaMalloc((void**)&dev_b, byte_size));
    CCE(cudaMalloc((void**)&dev_result, byte_size));

    CCE(cudaMemcpy(dev_a, a, byte_size, cudaMemcpyHostToDevice));
    CCE(cudaMemcpy(dev_b, b, byte_size, cudaMemcpyHostToDevice));

    int BLOCK_SIZE = 64;
    int NUMBER_OF_BLOCKS = size / BLOCK_SIZE + 1;
    
    time_stamp_reset()
    test<<<NUMBER_OF_BLOCKS, BLOCK_SIZE>>>(dev_a, dev_b, dev_result, size);
    CCE(cudaDeviceSynchronize());
    time_stamp("GPU");

    CCE(cudaMemcpy(result, dev_result, byte_size, cudaMemcpyDeviceToHost));
    CCE(cudaFree(dev_a));
    CCE(cudaFree(dev_b));
    CCE(cudaFree(dev_result));
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    // UTILS::clear_terminal();
    time_stamp("It just works");

    // OpenMP_GPU_test();

    MovieWriter movie_writer("random_pixels.mp4", def_WIDTH, def_HEIGHT, 1);
    vector<uint8_t> frame_buffer(4 * def_WIDTH * def_HEIGHT);
    memset(frame_buffer.data(), 0, 4 * def_WIDTH * def_HEIGHT);

    G::PIXEL_ARRAY_SIZE = def_PIXEL_ARRAY_SIZE;
    Renderer render;
    RGB* output{};



    for(int i=0; i<4; i++)
    {
        render.RENDER();
        output = render.get_my_pixel();

    }
    
    

    



    for (unsigned int j = 0; j < def_HEIGHT; j++)
        for (unsigned int i = 0; i < def_WIDTH; i++)
        {
            frame_buffer[4 * def_WIDTH * j + 4 * i + 2] = 0;        // red
            frame_buffer[4 * def_WIDTH * j + 4 * i + 1] = 255;      // green
            frame_buffer[4 * def_WIDTH * j + 4 * i + 0] = 0;        // blue
        }

    movie_writer.addFrame(&frame_buffer[0]);


    return 0;
}
