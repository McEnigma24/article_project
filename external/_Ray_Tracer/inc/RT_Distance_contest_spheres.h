#pragma once
#include "RT_Dimensions.h"
#include "RT_Sphere.h"
#include "base/_preprocessor_.h"

class Distance_contest_spheres
{
    u64 WIDTH;
    u64 HEIGHT;

    u8 counter;
    unit current_closest_distance;
    Hit_sphere current_closest_hit_sphere;

public:
    GPU_LINE(__host__ __device__)
    Distance_contest_spheres() : WIDTH(0), HEIGHT(0), counter(0), current_closest_distance(-1), current_closest_hit_sphere(false) {}

    GPU_LINE(__host__ __device__)
    static void init(Distance_contest_spheres& obj, u64 _WIDTH, u64 _HEIGHT)
    {
        obj.WIDTH = _WIDTH;
        obj.HEIGHT = _HEIGHT;
        obj.counter = 0;
        obj.current_closest_distance = -1;
        Hit_sphere::init(obj.current_closest_hit_sphere);
    }

    // juz wczesniej jest sprawdzane czy mamy trafienie czy nie
    GPU_LINE(__host__ __device__)
    void new_sphere(const Hit_sphere& hit_sphere)
    {
        if (counter == 0)
        {
            counter++;
            Hit_sphere::assign(current_closest_hit_sphere, hit_sphere);
        }
        else
        {
            // liczymy po raz pierwszy dystans do wczeœniej na œlepo wziêtego punktu przeciêcia
            if (counter == 1) { current_closest_distance = d3::distance_between(def_CAM_POS, current_closest_hit_sphere.intersection_pos); }

            unit distance_to_new_sphere = d3::distance_between(def_CAM_POS, hit_sphere.intersection_pos);

            if (distance_to_new_sphere < current_closest_distance)
            {
                counter++;
                current_closest_distance = distance_to_new_sphere;
                Hit_sphere::assign(current_closest_hit_sphere, hit_sphere);
            }
        }
    }

    GPU_LINE(__host__ __device__)
    void get_closest_sphere(Hit_sphere& ret) { Hit_sphere::assign(ret, current_closest_hit_sphere); }
};