#pragma once
#include "CTRL_Scene_details.h"
#include "CTRL_Scene_exe_details.h"
#include "RT_Light_point.h"
#include "RT_Sphere.h"
#include "base/_preprocessor_.h"

class Scene
{
    bool first_run;
    string name;

    // SCENE DATA
    vector<Light_point> light_points;
    vector<Sphere> spheres;

    // SCENE DETAILS
    u16 current_details_tab_index;
    vector<details> scene_obj_specifics;

    // EXECUTION DETAILS
    u16 current_thread_tab_index;
    vector<exe_details> scene_exe_specifics;

    void log_thread_details();
    void log_scene_details();

public:
    Scene();
    void assign_name(const string& _name);
    bool no_details_added();
    bool scene_empty();
    bool check_if_first_run();

    string get_name();

    u64 how_many_details_scene();
    u64 how_many_details_exe();

    u16 get_details_exe_num_of_threads();
    u16 get_details_exe_schema();
    u16 get_details_exe_block_size();

    vector<Light_point>& get_lights();
    vector<Sphere>& get_spheres();
    vector<details>& get_details();

    Light_point* get_lights_ptr();
    Sphere* get_spheres_ptr();
    details* get_details_ptr();

    GPU_LINE(__host__ __device__) u16 get_details_scene_spheres(GPU_LINE(details* details_tab))
    {
        GPU_LINE(return details_tab[current_details_tab_index].for_spheres);

        return scene_obj_specifics[current_details_tab_index].for_spheres;
    }
    GPU_LINE(__host__ __device__) u16 get_details_scene_lights(GPU_LINE(details* details_tab))
    {
        GPU_LINE(return details_tab[current_details_tab_index].for_lights);

        return scene_obj_specifics[current_details_tab_index].for_lights;
    }
    GPU_LINE(__host__ __device__) u16 get_details_scene_bounces(GPU_LINE(details* details_tab))
    {
        GPU_LINE(return details_tab[current_details_tab_index].for_bounces);

        return scene_obj_specifics[current_details_tab_index].for_bounces;
    }

    // TO POD KONIEC -> jezli false -> to next_iteration()
    bool is_SCENE_COMPLETED();
    void next_settings();

    void add_scene_detail(u16 lights, u16 spheres, u16 bounces);

    void add_thread_group(u16 number_of_threads_in_a_group, u16 schema, u16 block_size);

    void add_light(const d3& pos, const RGB& light_color);

    void add_sphere(const d3& c_pos, const unit& rad, const unit& transparent, const unit& reflective, Surface_type s_type, const RGB& col);

    friend class Renderer;
};