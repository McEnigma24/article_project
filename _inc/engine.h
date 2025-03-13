#include "sim.h"
// clang-format off

class Engine
{
    Computation_Box computation_box;
    INPUT_scene_OUTPUT_movie movie;

public:
    Engine(u64 _WIDTH, u64 _HEIGHT, const string& _name, int _frame_rate = 1) : movie(_WIDTH, _HEIGHT, _name, _frame_rate) {}

    void start(const sim_unit _space_WIDTH, const sim_unit _space_HEIGHT, const sim_unit _space_DEPTH, u64 number_of_iterations = 25)
    {
        Nano_Timer::Timer timer_SIM;
        int i = 0;

        #if defined(CPU) && defined(D_BUFF_DIFF_OBJ)
        {
            number_of_iterations += 2;
            i += 2;
        }
        #endif

        #if defined(CPU) && defined(D_BUFF_SAME_OBJ)
        {
            number_of_iterations += 1;
            i += 1;
        }
        #endif

        computation_box.fill_space_with_spheres(_space_WIDTH, _space_HEIGHT, _space_DEPTH, number_of_iterations);
        timer_SIM.start();
        {
            #if defined(CPU)
            {
                computation_box.cpu(number_of_iterations);
            }
            #endif

            #if defined(GPU)
            {
                computation_box.gpu(number_of_iterations);
            }
            #endif
        }
        timer_SIM.stop();

        #ifdef RENDER_ACTIVE
            time_stamp("Rendering started...");
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
            time_stamp("Rendering finished");

            movie.combine_to_movie();
            time_stamp("combine_to_movie");
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
