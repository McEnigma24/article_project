#pragma once
#include "CTRL_STATS_work_group.h"
#include "_preprocessor_.h"

class Stat_Record
{
    // SCENE DETAILS
    string scene_name;
    u16 scene_how_many_lights;
    u16 scene_how_many_spheres;
    u16 scene_how_many_bounces;

    // EXECUTION DETAILS
    u16 exe_how_many_threads;
    u16 exe_parallel_schema_index;
    u16 exe_block_size;

    work_group exe_time_report;

public:
    Stat_Record();
    Stat_Record(const string& _scene_name, u16 _scene_how_many_lights, u16 _scene_how_many_spheres,
                u16 _scene_how_many_bounces, u16 _exe_parallel_schema_index, u16 _exe_how_many_threads);

    work_group& get_stats();

    void save(string& file_output);
};