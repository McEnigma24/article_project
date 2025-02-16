#include "__preprocessor__.h"
#include "__time_stamp__.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "movie.h"

#include "CTRL_Scene.h"
#include "CTRL_Setuper.h"
#include "RT_Renderer.h"

#define CCE(x)                                                                                                         \
    {                                                                                                                  \
        cudaError_t err = x;                                                                                           \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            const string error = "CUDA ERROR - " + std::to_string(__LINE__) + " : " + __FILE__ + "\n";                 \
            cout << error;                                                                                             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }

__global__ void test(int* a, int* b, int* result, int ARRAY_SIZE)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (!(i < ARRAY_SIZE))
        return;

    result[i] = a[i] + b[i];
}

void OpenMP_GPU_test()
{
    int size = 1000000;
    int* a = new int[size];
    int* b = new int[size];
    int* result = new int[size];

    for (int i = 0; i < size; i++)
    {
        a[i] = i;
        b[i] = size - i;
    }

    time_stamp_reset();

    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] + b[i];
    }
    time_stamp("Iterative");

#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++)
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

    time_stamp_reset() test<<<NUMBER_OF_BLOCKS, BLOCK_SIZE>>>(dev_a, dev_b, dev_result, size);
    CCE(cudaDeviceSynchronize());
    time_stamp("GPU");

    CCE(cudaMemcpy(result, dev_result, byte_size, cudaMemcpyDeviceToHost));
    CCE(cudaFree(dev_a));
    CCE(cudaFree(dev_b));
    CCE(cudaFree(dev_result));
}

// unit z_plane = u(100);

// scene.add_light
// (
//     d3(u(G::WIDTH / 2), u(G::HEIGHT / 2), z_plane)
//     ,RGB(255, 255, 255)
// );

// scene.add_sphere
// (
//     d3(u(G::WIDTH / 2), u(G::HEIGHT / 2), z_plane), u(100),		// DEAD CENTER
//     0.0f, 0.0f, Surface_type::diffuse,
//     RGB(0, 255, 0)
// );

// #define def_WIDTH (1024)
// #define def_HEIGHT (768)

#define def_WIDTH (1000)
#define def_HEIGHT (1000)
#define def_convert_2d_to_1d(x, y) (y * def_WIDTH + x)

#define FRAMES (1)

class Movie_Maker_Controller
{
    vector<uint8_t> frame_buffer;
    vector<vector<RGB>> saved_frames;

    void fill_frame_buffer(const vector<RGB>& render_output)
    {
        for (unsigned int y = 0; y < def_HEIGHT; y++)
            for (unsigned int x = 0; x < def_WIDTH; x++)
            {
                frame_buffer[4 * def_WIDTH * y + 4 * x + 2] = render_output[def_convert_2d_to_1d(x, y)].get_r();
                frame_buffer[4 * def_WIDTH * y + 4 * x + 1] = render_output[def_convert_2d_to_1d(x, y)].get_g();
                frame_buffer[4 * def_WIDTH * y + 4 * x + 0] = render_output[def_convert_2d_to_1d(x, y)].get_b();
            }
    }

public:
    Movie_Maker_Controller() : frame_buffer(4 * def_WIDTH * def_HEIGHT) {}

    void add_new_frame(const vector<RGB>& frame) { saved_frames.push_back(frame); }

    void combine_to_movie(const string& name, int frame_rate = 1)
    {
        MovieWriter movie_writer(name, def_WIDTH, def_HEIGHT, frame_rate);

        // int how_many_added_frames{};
        // for (int i = 0; i < saved_frames.size(); i++)
        // {
        //     memset(frame_buffer.data(), 0, 4 * def_WIDTH * def_HEIGHT);
        //     fill_frame_buffer(saved_frames[i]);

        //     if (i == 0 || i == saved_frames.size() - 1)
        //         how_many_added_frames = 30;
        //     else
        //         how_many_added_frames = 5;

        //     for (int ii; ii < FRAMES * how_many_added_frames; ii++)
        //         movie_writer.addFrame(&frame_buffer[0]);
        // }
    }

    void delete_all_collected_frames() { saved_frames.clear(); }
};

void fill_frame_buffer(const vector<RGB>& render_output, vector<u8>& frame_buffer)
{
    for (unsigned int y = 0; y < def_HEIGHT; y++)
        for (unsigned int x = 0; x < def_WIDTH; x++)
        {
            frame_buffer[4 * def_WIDTH * y + 4 * x + 2] = render_output[def_convert_2d_to_1d(x, y)].get_r();
            frame_buffer[4 * def_WIDTH * y + 4 * x + 1] = render_output[def_convert_2d_to_1d(x, y)].get_g();
            frame_buffer[4 * def_WIDTH * y + 4 * x + 0] = render_output[def_convert_2d_to_1d(x, y)].get_b();
        }
}

void crash_function()
{
    int* a{};
    *a = 0;
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    time_stamp("It just works");
    Movie_Maker_Controller maker;

    crash_function();

    maker.combine_to_movie("my_maker.mp4");

    return 0;

    Setuper::setup_Global_Variables___and___Clear_Stats();
    Renderer render(def_WIDTH, def_HEIGHT);

    if (false)
    {
        MovieWriter movie_writer("random_pixels.mp4", def_WIDTH, def_HEIGHT, 2);
        vector<uint8_t> frame_buffer(4 * def_WIDTH * def_HEIGHT);
        memset(frame_buffer.data(), 0, 4 * def_WIDTH * def_HEIGHT);

        Setuper::setup_Global_Variables___and___Clear_Stats();
        Renderer render(def_WIDTH, def_HEIGHT);

        {
            Scene scene;
            Setuper::setup_scene_0(&scene, "first");
            G::Render::current_scene = &scene;
            render.RENDER();
            line("rendering");

            fill_frame_buffer(render.get_my_pixel_vec(), frame_buffer);
            for (int ii; ii < FRAMES * 30; ii++)
                movie_writer.addFrame(&frame_buffer[0]);
        }

        {
            Scene scene;
            Setuper::setup_scene_1(&scene, "first");
            G::Render::current_scene = &scene;
            render.RENDER();
            line("rendering");

            fill_frame_buffer(render.get_my_pixel_vec(), frame_buffer);
            for (int ii; ii < FRAMES * 5; ii++)
                movie_writer.addFrame(&frame_buffer[0]);
        }

        {
            Scene scene;
            Setuper::setup_scene_2(&scene, "first");
            G::Render::current_scene = &scene;
            render.RENDER();
            line("rendering");

            fill_frame_buffer(render.get_my_pixel_vec(), frame_buffer);
            for (int ii; ii < FRAMES * 5; ii++)
                movie_writer.addFrame(&frame_buffer[0]);
        }

        movie_writer.addFrame(&frame_buffer[0]);
        {
            Scene scene;
            Setuper::setup_scene_6(&scene, "first");
            G::Render::current_scene = &scene;
            render.RENDER();
            line("rendering");

            fill_frame_buffer(render.get_my_pixel_vec(), frame_buffer);
            for (int ii; ii < FRAMES * 5; ii++)
                movie_writer.addFrame(&frame_buffer[0]);
        }

        {
            Scene scene;
            Setuper::setup_scene_3(&scene, "first");
            G::Render::current_scene = &scene;
            render.RENDER();
            line("rendering");

            fill_frame_buffer(render.get_my_pixel_vec(), frame_buffer);
            for (int ii; ii < FRAMES * 30; ii++)
                movie_writer.addFrame(&frame_buffer[0]);
        }
    }

    if (false)
    {
        {
            Scene scene;
            Setuper::setup_scene_0(&scene, "first");
            G::Render::current_scene = &scene;

            render.RENDER();
            maker.add_new_frame(render.get_my_pixel_vec());
            line("here");
        }

        {
            Scene scene;
            Setuper::setup_scene_1(&scene, "first");
            G::Render::current_scene = &scene;

            render.RENDER();
            maker.add_new_frame(render.get_my_pixel_vec());
        }

        {
            Scene scene;
            Setuper::setup_scene_2(&scene, "first");
            G::Render::current_scene = &scene;

            render.RENDER();
            maker.add_new_frame(render.get_my_pixel_vec());
        }

        {
            Scene scene;
            Setuper::setup_scene_3(&scene, "first");
            G::Render::current_scene = &scene;

            render.RENDER();
            maker.add_new_frame(render.get_my_pixel_vec());
        }

        line("here 1");
        maker.combine_to_movie("my_maker.mp4");
        line("here 2");
        maker.delete_all_collected_frames();
        line("here 3");
    }

    return 0;
}
