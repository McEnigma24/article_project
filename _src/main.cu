#include "__preprocessor__.h"
#include "__time_stamp__.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "movie.h"

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

    const unsigned int width = 1024;
    const unsigned int height = 768;
    const unsigned int nframes = 64;

    MovieWriter movie_writer("random_pixels.mp4", width, height, 1);

    vector<uint8_t> pixels(4 * width * height);
	for (unsigned int iframe = 0; iframe < nframes; iframe++)
	{
		for (unsigned int j = 0; j < height; j++)
			for (unsigned int i = 0; i < width; i++)
			{
				pixels[4 * width * j + 4 * i + 0] = 0;        // blue
				pixels[4 * width * j + 4 * i + 1] = 255;        // green
				pixels[4 * width * j + 4 * i + 2] = 0;        // red
				pixels[4 * width * j + 4 * i + 3] = 255;        // alpha
			}	

            movie_writer.addFrame(&pixels[0]);
	}

    return 0;
}
