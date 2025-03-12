#include "sim.h"

// whole iteration after iteration -> rendering

class Engine
{
    Computation_Box computation_box;
    INPUT_scene_OUTPUT_movie movie;

public:
    Engine(u64 _WIDTH, u64 _HEIGHT, const string& _name, int _frame_rate = 1) : movie(_WIDTH, _HEIGHT, _name, _frame_rate) {}

    void start(const unit _space_WIDTH, const unit _space_HEIGHT, const unit _space_DEPTH, u64 number_of_iterations = 25)
    {
        Nano_Timer::Timer timer_SIM;

        int i = 0;

// double //
#ifdef NEXT_VALUE_IN_SAME_OBJ___ONLY_FOR_D_BUFFERING
        number_of_iterations += 1;
        i += 1;
#else
        number_of_iterations += 2;
        i += 2;
#endif

        computation_box.fill_space_with_spheres(_space_WIDTH, _space_HEIGHT, _space_DEPTH, number_of_iterations);
        timer_SIM.start();
        {
            // computation_box.cpu_N_buffering(number_of_iterations);
            computation_box.cpu_double_buffering(number_of_iterations);
        }
        timer_SIM.stop();

#ifdef RENDER_ACTIVE

        Nano_Timer::Timer timer_Ray_Tracing;
        timer_Ray_Tracing.start();
        {
            for (; i < number_of_iterations; i++)
            {
                Scene scene;
                computation_box.transform_to_My_Ray_Tracing_scene(scene, i);
                movie.add_scene(scene);
            }
        }
        timer_Ray_Tracing.stop();

        movie.combine_to_movie();

#endif

        nline;
        line("timer_SIM");
        timer_SIM.log();
        nline;
        line("timer_Ray_Tracing");
        timer_Ray_Tracing.log();
        nline;
    }
};
