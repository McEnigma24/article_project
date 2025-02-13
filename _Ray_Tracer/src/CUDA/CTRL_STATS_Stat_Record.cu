#include "CTRL_STATS_Stat_Record.h"
#include "CTRL_Scene.h"
#include <tuple>
#include <sstream>
#include "_MAIN_.h"



Stat_Record::Stat_Record(const string& _scene_name, u16 _scene_how_many_lights, u16 _scene_how_many_spheres,
    u16 _scene_how_many_bounces, u16 _exe_parallel_schema_index, u16 _exe_how_many_threads)
    : c_init(scene_name), c_init(scene_how_many_lights), c_init(scene_how_many_spheres), c_init(scene_how_many_bounces), c_init(exe_parallel_schema_index), c_init(exe_how_many_threads),
    exe_time_report(_exe_how_many_threads)
{
}

Stat_Record::Stat_Record()
    : exe_time_report(G::Render::current_scene->get_details_exe_num_of_threads())
{
    auto& scene = *G::Render::current_scene;

    scene_name = scene.get_name();
    scene_how_many_lights = scene.get_details_scene_lights(GPU_LINE(G::Render::current_scene->get_details_ptr()));
    scene_how_many_spheres = scene.get_details_scene_spheres(GPU_LINE(G::Render::current_scene->get_details_ptr()));
    scene_how_many_bounces = scene.get_details_scene_bounces(GPU_LINE(G::Render::current_scene->get_details_ptr()));
    
    exe_parallel_schema_index = scene.get_details_exe_schema();
    exe_block_size = scene.get_details_exe_block_size();
    exe_how_many_threads = scene.get_details_exe_num_of_threads();
}

work_group& Stat_Record::get_stats()
{
    return exe_time_report;
}

/////////////////////////////////////////////////////////////////////////////

#include "_PARSER_Format_Buffer_Common_IMP_.h"

/////////////////////////////////////////////////////////////////////////////

void Stat_Record::save(string& output)
{
    #ifdef CPU
    stringstream stream;
    stream << fixed << setprecision(2) << (float)omp_get_num_procs() / (float)stoi(G::NUM_OF_CORES);
    #endif

    auto [min, avg, max, rel_dev] = exe_time_report.get_whole();

    tupl variables =
    {
        rel_dev,
        min,
        avg,
        max,

        CPU_LINE("CPU")
        GPU_LINE("GPU")
        ,

        G::MODEL_NAME,

        ARCH_X86_LINE("x86")
        ARCH_X64_LINE("x64")
        GPU_LINE("CUDA")
        ,

        UNIT_FLOAT_LINE("float")
        UNIT_DOUBLE_LINE("double")
        ,

        G::NUM_OF_CORES,

        CPU_LINE(to_string(omp_get_num_procs()))
        GPU_LINE(to_string(0))
        ,

        CPU_LINE(stream.str())
        GPU_LINE(to_string(0))
        ,

        scene_name,
        to_string(scene_how_many_lights),
        to_string(scene_how_many_spheres),
        to_string(scene_how_many_bounces),
        
        CPU_LINE(to_string(exe_how_many_threads))
        GPU_LINE(to_string(0))
        ,
        
        CPU_LINE("measuring")
        GPU_LINE("BEST")
    };

    // normal log
    output += Format_Buffer::input_variables_return_log_line(variables);
    insert_raw_datapoints(output, exe_time_report.get_whole_data());    
    output += "\n";

    #ifdef CPU

    static vector<u64> best_data;
    static tupl best_time_variables;
    static u64 best_time_so_far = -1; // biggest uint64_t

    if(exe_how_many_threads != def_MAX_NUM_OF_THREADS)
    {
        if(avg < best_time_so_far)
        {
            best_data = exe_time_report.get_whole_data();
            best_time_so_far = avg;
            best_time_variables = variables;
        }
    }
    else
    {
        auto& rating = GET(best_time_variables, RATING + 4);
        rating = "BEST";

        // additional BEST log
        output += Format_Buffer::input_variables_return_log_line(best_time_variables);
        insert_raw_datapoints(output, best_data);
        output += "\n";

        // RESETING //
        best_time_so_far = -1;
    }
    #endif

    
    // // output += " -> Timers single(";
    // //  1:(time), 2:(time) ...
    // // 
    // //  output += exe_time_report.get_single();
    // // output += ") ";
}