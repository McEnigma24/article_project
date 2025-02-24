#pragma once
#include "RT_Hit_info_and_position.h"
#include "RT_Ray.h"
#include "base/_preprocessor_.h"

struct Hit_sphere : public Hit_info_and_position
{
    Ray normal;
    u64 sphere_index;

    GPU_LINE(__host__ __device__)
    static void init(Hit_sphere& hit)
    {
        hit.intersecting_with_object = false;

        hit.intersection_pos.x = u(0);
        hit.intersection_pos.y = u(0);
        hit.intersection_pos.z = u(0);
    }

    GPU_LINE(__host__ __device__)
    static void assign(Hit_sphere& hit, const Hit_sphere& other)
    {
        member_assign(hit, other, sphere_index);
        member_assign(hit, other, intersecting_with_object);

        member_assign(hit, other, intersection_pos.x);
        member_assign(hit, other, intersection_pos.y);
        member_assign(hit, other, intersection_pos.z);
    }

    GPU_LINE(__host__ __device__)
    Hit_sphere(bool b = false, const d3& int_pos = -1) : Hit_info_and_position(b, int_pos), normal(), sphere_index(-1) {}

    GPU_LINE(__host__ __device__)
    void add_sphere_index(const u64& _sphere_index) { sphere_index = _sphere_index; }

    GPU_LINE(__host__ __device__)
    void add_normal__provide_sphere_center(const d3& _center_pos) { normal.modify_after_construction(_center_pos, intersection_pos); }

    GPU_LINE(__host__ __device__)
    Hit_sphere& operator=(const Hit_sphere& other)
    {
        if (this != &other)
        {
            THIS_OTHER(intersecting_with_object);
            THIS_OTHER(intersection_pos);
            THIS_OTHER(normal);
            THIS_OTHER(sphere_index);
        }
        return *this;
    }

#ifdef CPU
    void print(const string& var_name) const
    {
        print_var_name;

        varr(intersecting_with_object);
        call_print(intersection_pos);
        var(sphere_index);
        call_print(normal);
    }
#endif
};