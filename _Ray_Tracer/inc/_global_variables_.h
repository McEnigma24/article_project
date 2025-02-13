#pragma once
#include "_constants_.h"
#include "_shortcut_logs_.h"
#include "_type_definitions_.h"

#include "RT_RGB_float.h"
#include <random>

struct d3;
class Scene;
class Stat_Record;

#define get_prefix(x) get_##x
#define fun_get(type, x)                                                                                               \
    static type get_prefix(x)() { return x; }
#define zero_to(x) x = 0;

const string path_rep_num = "../_run_time_config_/rep_number.txt";

struct G
{
    static u64 REP_NUMBER;

    static u64 WIDTH;
    static u64 HEIGHT;
    static u64 PIXEL_ARRAY_SIZE;

    static unit EPSILON;
    static u16 PIXEL_SKIP;

    static string MODEL_NAME;
    static string NUM_OF_CORES;

    static float SCALING_MULTI;
    static float SCALING_ADD;

    static u64 PROGRESS_COUNTER;
    static u64 PROGRESS_ALL;

    enum ray_type
    {
        camera_point,
        parallel
    };
    static ray_type RAY_TYPE;

    struct Camera
    {
        static d3 pos;

        static d3 move_offset;
        static unit move_speed;

        static d3 rotation_angles;
        static unit rotation_speed;

        static unit pow2_pos_sum;

        static d3 starting_ray;

        static unit inc_x;
        static unit inc_y;

        static unit distance(const d3& other_pos);
    };

    struct Render
    {
        static Scene* current_scene;

        static Stat_Record* current_scene_stats;
    };
};
