#pragma once
#include "FUNC_Float_functions.h"
#include "RT_Dimensions.h"
#include "RT_Hit_sphere.h"
#include "RT_Object.h"
#include "RT_Ray.h"
#include "_preprocessor_.h"

struct Sphere : public Object
{
    d3 center_pos;
    unit radious;

public:
    GPU_LINE(__host__ __device__)
    Sphere(d3 c_pos, unit rad, unit transparent, unit reflective, bool light, RGB_float col, Surface_type s_type)
        : Object(transparent, reflective, light, col, s_type), center_pos(c_pos), radious(rad)
    {
    }

    GPU_LINE(__host__ __device__)
    void intersection(Hit_sphere& hit, const Ray& ray)
    {
        CPU_LINE(SAFETY_CHECK(ASSERT_ER_IF_TRUE(ray.uninitialized_values)));

        unit a, b, c, delta;

        a = pow2(ray.d.x) + pow2(ray.d.y) + pow2(ray.d.z);

        b = 2 * (ray.d.x * (ray.s.x - center_pos.x) + ray.d.y * (ray.s.y - center_pos.y) +
                 ray.d.z * (ray.s.z - center_pos.z));

        c = pow2(center_pos.x) + pow2(center_pos.y) + pow2(center_pos.z) +

            pow2(ray.s.x) + pow2(ray.s.y) + pow2(ray.s.z) - pow2(radious)

            - 2 * ((center_pos.x * ray.s.x) + (center_pos.y * ray.s.y) + (center_pos.z * ray.s.z));
        ;

        delta = (b * b) - 4 * (a * c);

        // no hit
        if (Floating_functions::comparison_smaller_than_zero(delta))
        {
            hit.intersecting_with_object = false;
            return;
        }

        CPU_LINE(SAFETY_CHECK(ASSERT_ER_IF_TRUE(a == 0)));
        CPU_LINE(SAFETY_CHECK(ASSERT_ER_IF_TRUE(Floating_functions::comparison_equal_to_zero(a))));

        unit t = ((-b - sqrt(delta)) / (2 * a));

        if (t < 0.1)
        {
            t = ((-b + sqrt(delta)) / (2 * a));
        }

        if (t < 0.01)
        {
            hit.intersecting_with_object = false;
            return;
        }

        hit.intersecting_with_object = true;
        hit.intersection_pos.x = (ray.s.x + (t * ray.d.x));
        hit.intersection_pos.y = (ray.s.y + (t * ray.d.y));
        hit.intersection_pos.z = (ray.s.z + (t * ray.d.z));
    }
};
