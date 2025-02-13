#pragma once
#include "CTRL_Parallel_CPU.h"
#include "CTRL_Scene.h"
#include "CTRL_Scene_Controller.h"
#include "RT_Light_calculations.h"
#include "RT_Ray.h"
#include "_preprocessor_.h"

class Renderer
{
    WIN_LINE(sf::RenderWindow& window);
    WIN_LINE(VertexArray sfml_pixels);
    CPU_LINE(Parallel_CPU parallel_controller);

    vector<RGB> my_pixel;
    Scene_Controller scene_controller;

public:
    Renderer(WIN_LINE(sf::RenderWindow& w));

    GPU_LINE(__host__ __device__)
    RGB* get_my_pixel() { return my_pixel.data(); }

    void RENDER();
    bool test_is_finished();
    void choose_first_scene(u16 index);

    void save_frame();

#ifdef WIN
    void setup_all_pixels();
    void reset_all_pixels();
    void copy_from_pixel_colors();
    void show_frame();
#endif

    WIN_LINE(void set_pixel_hack_for_showing_schema(const u64& coord, const Color& color);)

    GPU_LINE(__host__ __device__)
    void per_pixel(const u64& coord, Scene* current_scene, RGB* my_pixel, Light_point* current_scene_lights,
                   Sphere* current_scene_spheres, details* current_scene_details);

    GPU_LINE(__host__ __device__)
    void per_pixel(const u64& x, const u64& y, Scene* current_scene, RGB* my_pixel, Light_point* current_scene_lights,
                   Sphere* current_scene_spheres, details* current_scene_details);

    GPU_LINE(__host__ __device__)
    void ray_looking_for_sphere(Hit_sphere& hit, const Ray& ray, Scene* current_scene,
                                Light_point* current_scene_lights, Sphere* current_scene_spheres,
                                details* current_scene_details);

    friend class Parallel_CPU;
};