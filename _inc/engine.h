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

        for (int i = 0; i < 10; i++)
        {
            Scene current_scene;
            // Setuper::setup_scene_0(&current_scene, "first");

            string current_i = "Iteration: " + to_string(i) + " ";

            // computation_box.temp_dist(memory_index.get());
            // time_stamp(current_i + "temp_dist");

            computation_box.collision_resolution(memory_index.get());
            time_stamp(current_i + "collision_resolution");

            computation_box.transform_to_My_Ray_Tracing_scene(current_scene);
            time_stamp(current_i + "transform_to_My_Ray_Tracing_scene");

            movie.add_scene(current_scene);
            time_stamp(current_i + "add_scene");

            memory_index.switch_to_next();

            line("end");
        }

        movie.combine_to_movie();
        time_stamp("combine_to_movie");
    }
};
