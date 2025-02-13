#pragma once
#include "CTRL_Scene.h"
#include "FUNC_Float_functions.h"
#include "RT_RGB_float_sum_and_avg.h"
#include "_preprocessor_.h"

struct Light_calculations
{
    GPU_LINE(__host__ __device__)
    static bool single_light_ray_blocked(const Ray& ray, Sphere* primary, Scene* current_scene,
                                         Light_point* current_scene_lights, Sphere* current_scene_spheres,
                                         details* current_scene_details)
    {
        Hit_sphere hit(false);

        const unit& distance_from_primary_intersectionn_to_light_source = ray.distance;
        unit distance_from_other_sphere_intersection_to_primary_intersection;

        CPU_LINE(SAFETY_CHECK(ASSERT_ER_IF_NULL((current_scene))));

        CPU_LINE(auto scene_all_spheres = (current_scene)->get_spheres());
        GPU_LINE(auto scene_all_spheres = current_scene_spheres);

        u16 limit = (current_scene)->get_details_scene_spheres(GPU_LINE(current_scene_details));

        for (u16 i = 0; i < limit; i++)
        {
            auto& sphere = scene_all_spheres[i];

            // LIGHT SOURCE can't block
            if (sphere.ligth_source)
            {
                continue;
            }

            if (&sphere != primary)
            {
                sphere.intersection(hit, ray);

                if (hit.intersecting_with_object)
                {
                    distance_from_other_sphere_intersection_to_primary_intersection =
                        d3::distance_between(ray.s, hit.intersection_pos);

                    if (!Floating_functions::comparison_bigger___true_if_a_bigger_than_b(
                            distance_from_other_sphere_intersection_to_primary_intersection,
                            distance_from_primary_intersectionn_to_light_source))
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    GPU_LINE(__host__ __device__)
    static unit intensity_based_on_angle(const Ray& light_ray, const Ray& normal)
    {
        unit a = light_ray.cos(normal);

        if (a < u(0))
            return u(0);
        return a;
    }

    GPU_LINE(__host__ __device__)
    static void finding_light_sources_average_color_weighted_with_angle_and_distance(
        RGB_float_sum_and_avg& combining_all_light_sources, const Hit_sphere& hit, Scene* current_scene,
        Light_point* current_scene_lights, Sphere* current_scene_spheres, details* current_scene_details)
    {
        CPU_LINE(SAFETY_CHECK(ASSERT_ER_IF_NULL((current_scene))));

        auto& hit_sphere = current_scene_spheres[hit.sphere_index];

        CPU_LINE(auto scene_all_lights = (current_scene)->get_lights());
        GPU_LINE(auto scene_all_lights = current_scene_lights);

        u16 limit = (current_scene)->get_details_scene_lights(GPU_LINE(current_scene_details));

        for (u16 i = 0; i < limit; i++)
        {
            auto& light_source = scene_all_lights[i];

            Ray ray_from_intersection_to_light_source;
            Ray::modify_after_construction(ray_from_intersection_to_light_source, hit.intersection_pos,
                                           light_source.pos);

            if (!single_light_ray_blocked(ray_from_intersection_to_light_source, &hit_sphere, current_scene,
                                          current_scene_lights, current_scene_spheres, current_scene_details))
            {
                RGB_float color_from_one_light;

                RGB_float::assign(color_from_one_light, light_source.light_color);
                RGB_float::mix_with_unit(color_from_one_light,
                                         intensity_based_on_angle(ray_from_intersection_to_light_source, hit.normal));

                RGB_float::mix_with_RGB_float(color_from_one_light, hit_sphere.color);
                RGB_float_sum_and_avg::add_color(combining_all_light_sources, color_from_one_light);
            }
        }
    }
};
