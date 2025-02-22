#include "sim.h"

// whole iteration after iteration -> rendering

class Engine
{
    Computation_Box computation_box;
    INPUT_scene_OUTPUT_movie movie;

public:
    Engine(unit _space_WIDTH, unit _space_HEIGHT, unit _space_DEPTH, u64 _WIDTH, u64 _HEIGHT, const string& _name, int _frame_rate = 1)
        : movie(_WIDTH, _HEIGHT, _name, _frame_rate)
    {
        computation_box.fill_space_with_spheres(_space_WIDTH, _space_HEIGHT, _space_DEPTH);
    }

    void start()
    {
        Memory_index memory_index;
        Nano_Timer::Timer timer_sim_Temp_Dirt;
        Nano_Timer::Timer timer_sim_Collision_Res;
        Nano_Timer::Timer timer_Scene_Creation;
        Nano_Timer::Timer timer_Ray_Tracing;

        for (int i = 0; i < 15; i++)
        {
            Scene current_scene;
            // Setuper::setup_scene_0(&current_scene, "first");

            string current_i = "Iteration: " + to_string(i) + " ";

            timer_sim_Collision_Res.start();
            {
                computation_box.iteration_step(memory_index);
                time_stamp(current_i + "collision_resolution");
            }
            timer_sim_Collision_Res.stop();

#ifdef RENDER
            timer_Scene_Creation.start();
            {
                computation_box.transform_to_My_Ray_Tracing_scene(current_scene, memory_index);
                time_stamp(current_i + "transform_to_My_Ray_Tracing_scene");
            }
            timer_Scene_Creation.stop();

            timer_Ray_Tracing.start();
            {
                movie.add_scene(current_scene);
                time_stamp(current_i + "add_scene");
            }
            timer_Ray_Tracing.stop();
#endif

            memory_index.switch_to_next();

            line("end");
        }

#ifdef RENDER
        movie.combine_to_movie();
        time_stamp("combine_to_movie");
#endif

        nline;
        line("timer_sim_Collision_Res");
        timer_sim_Collision_Res.log();
        nline;
        line("timer_Scene_Creation");
        timer_Scene_Creation.log();
        nline;
        line("timer_Ray_Tracing");
        timer_Ray_Tracing.log();
        nline;
    }
};
