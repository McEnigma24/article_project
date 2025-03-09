#include "__preprocessor__.h"
#include "__time_stamp__.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "engine.h"
#include "openMP_test.h"

int main(int argc, char* argv[])
{
    srand(time(NULL));

    // show_sizeof_many(Sim_sphere, 36000 * 50);

    unit times = u(1);

    // Engine engine(sm * u(40), sm * u(100), sm * u(20), // sim space params
    Engine engine(1000, 1000,    // render params
                  "movie.mp4", 5 // movie params
    );
    engine.start(times * u(120), times * u(220), times * u(30), // sim space params
                 25);

    // OpenMP_GPU_test();

    return 0;
}
