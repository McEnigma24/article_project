#include "__preprocessor__.h"
#include "__time_stamp__.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "openMP_test.h"

#define COLLISION_RESOLUTION
#define TEMP_DISTRIBUTION
#define VOLUME_TRANSFER
#include "engine.h"

string tytle()
{
    string ret;

#if defined(CPU) && defined(N_BUFF)
    {
        ret = "CPU-N-BUFF.mp4";
    }
#endif

#if defined(CPU) && defined(D_BUFF_DIFF_OBJ)
    {
        ret = "CPU-D-BUFF-DIFF-OBJ.mp4";
    }
#endif

#if defined(CPU) && defined(D_BUFF_SAME_OBJ)
    {
        ret = "CPU-D-BUFF-SAME-OBJ.mp4";
    }
#endif

#if defined(GPU) && defined(N_BUFF)
    {
        ret = "GPU-N-BUFF.mp4";
    }
#endif

    return ret;
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    // show_sizeof_many(Sim_sphere, 36000 * 50);

    string movie_tytle = tytle();

    unit times = sim_u(1);

    // Engine engine(sm * sim_u(40), sm * sim_u(100), sm * sim_u(20), // sim space params
    Engine engine(1000, 1000,    // render params
                  movie_tytle, 5 // movie params
    );
    engine.start(times * sim_u(120), times * sim_u(220), times * sim_u(30), // sim space params
                 25);

    // OpenMP_GPU_test();

    return 0;
}
