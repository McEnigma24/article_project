#include "sim.h"

// whole iteration after iteration -> rendering

class Engine
{
    Computation_Box computation_box;
    INPUT_scene_OUTPUT_movie movie;

public:
    Engine(u64 _WIDTH, u64 _HEIGHT, const string& _name, int _frame_rate = 1) : movie(_WIDTH, _HEIGHT, _name, _frame_rate) {}

    void start(const unit _space_WIDTH, const unit _space_HEIGHT, const unit _space_DEPTH, const u64 number_of_iterations = 25)
    {
        Nano_Timer::Timer timer_SIM;
        Nano_Timer::Timer timer_Scene_Creation;
        Nano_Timer::Timer timer_Ray_Tracing;

        computation_box.fill_space_with_spheres(_space_WIDTH, _space_HEIGHT, _space_DEPTH, number_of_iterations);

        timer_SIM.start();
        {
            computation_box.;
            time_stamp(current_i + "collision_resolution");
        }
        timer_SIM.stop();

        //         for (int i = 0; i < 19; i++)
        //         {

        //             string current_i = "Iteration: " + to_string(i) + " ";

        //             timer_SIM.start();
        //             {
        //                 computation_box.iteration_step(memory_index);
        //                 time_stamp(current_i + "collision_resolution");
        //             }
        //             timer_SIM.stop();

        // #ifdef RENDER_ACTIVE
        //             Scene current_scene;
        //             timer_Scene_Creation.start();
        //             {
        //                 computation_box.transform_to_My_Ray_Tracing_scene(current_scene, memory_index);
        //                 time_stamp(current_i + "transform_to_My_Ray_Tracing_scene");
        //             }
        //             timer_Scene_Creation.stop();

        //             timer_Ray_Tracing.start();
        //             {
        //                 movie.add_scene(current_scene);
        //                 time_stamp(current_i + "add_scene");
        //             }
        //             timer_Ray_Tracing.stop();
        // #endif

        //             memory_index.switch_to_next();

        //             line("end");
        //         }

        // #ifdef RENDER_ACTIVE
        //         movie.combine_to_movie();
        //         time_stamp("combine_to_movie");
        // #endif

        nline;
        line("timer_SIM");
        timer_SIM.log();
        nline;
        line("timer_Scene_Creation");
        timer_Scene_Creation.log();
        nline;
        line("timer_Ray_Tracing");
        timer_Ray_Tracing.log();
        nline;
    }
};
