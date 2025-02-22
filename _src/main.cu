#include "__preprocessor__.h"
#include "__time_stamp__.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "engine.h"
#include "openMP_test.h"

int main(int argc, char* argv[])
{
    srand(time(NULL));

    // proporcje obszaru obliczeniowego
    // 10x * 4x * x

    // r = 10;
    // x = 100;
    //           -> 40'000 sfer

    // cout << 35937 * sizeof(Sim_sphere) << endl;

    // Engine engine(sm * u(40), sm * u(100), sm * u(20), // sim space params
    Engine engine(u(200), u(200), u(200), // sim space params
                                          // 1024, 768,                     // render params
                  1000, 1000,             // render params
                  "movie.mp4", 5          // movie params
    );
    engine.start();

    // OpenMP_GPU_test();

    return 0;
}
