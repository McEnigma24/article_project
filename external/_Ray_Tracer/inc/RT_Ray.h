#pragma once
#include "FUNC_Random_functions.h"
#include "RT_Dimensions.h"
#include "_preprocessor_.h"

struct Ray
{
    d3 s; // starting position
    d3 d; // direction
    unit distance;
    bool uninitialized_values;

public:
#define ray_prep                                                                                                       \
    {                                                                                                                  \
        d.x = ending_pos.x - starting_pos.x;                                                                           \
        d.y = ending_pos.y - starting_pos.y;                                                                           \
        d.z = ending_pos.z - starting_pos.z;                                                                           \
                                                                                                                       \
        distance = d3::distance(d);                                                                                    \
                                                                                                                       \
        d.x /= distance;                                                                                               \
        d.y /= distance;                                                                                               \
        d.z /= distance;                                                                                               \
    }

    GPU_LINE(__host__ __device__)
    unit dot_prod(const Ray& other) const
    {
        CPU_LINE(SAFETY_CHECK(ASSERT_ER_IF_TRUE(uninitialized_values)));

        return ((this->d.x * other.d.x) + (this->d.y * other.d.y) + (this->d.z * other.d.z));
    }

    GPU_LINE(__host__ __device__)
    static unit dot_prod(const Ray& r1, const Ray& r2)
    {
        CPU_LINE(SAFETY_CHECK(ASSERT_ER_IF_TRUE(r1.uninitialized_values)));
        CPU_LINE(SAFETY_CHECK(ASSERT_ER_IF_TRUE(r2.uninitialized_values)));

        return r1.dot_prod(r2);
    }

    GPU_LINE(__host__ __device__)
    unit angle(const Ray& other) const
    {
        CPU_LINE(SAFETY_CHECK(ASSERT_ER_IF_TRUE(uninitialized_values)));

        unit cos_x = (this->dot_prod(other));

        UNIT_FLOAT_LINE(unit ret = acosf(cos_x));
        UNIT_DOUBLE_LINE(unit ret = acosl(cos_x));

        return ret;
    }

    GPU_LINE(__host__ __device__)
    unit angle(const Ray& r1, const Ray& r2) { return r1.angle(r2); }

    GPU_LINE(__host__ __device__)
    unit cos(const Ray& other) const
    {
        CPU_LINE(SAFETY_CHECK(ASSERT_ER_IF_TRUE(uninitialized_values)));

        unit ret = (this->dot_prod(other));

        return ret;
    }

    GPU_LINE(__host__ __device__)
    unit cos(const Ray& r1, const Ray& r2) { return r1.cos(r2); }

    GPU_LINE(__host__ __device__)
    Ray() : s(), uninitialized_values(true) {}

    GPU_LINE(__host__ __device__)
    Ray(const d3& starting_pos, const d3& ending_pos) : s(starting_pos), uninitialized_values(false) { ray_prep; }

    GPU_LINE(__host__ __device__)
    Ray& operator=(const Ray& other)
    {
        THIS_OTHER(s);
        THIS_OTHER(d);
        THIS_OTHER(distance);
        THIS_OTHER(uninitialized_values);

        return *this;
    }

    GPU_LINE(__host__ __device__)
    static void assign(Ray& main, const Ray& other)
    {
        d3::assign(main.s, other.s);
        d3::assign(main.d, other.d);

        member_assign(main, other, distance);
        member_assign(main, other, uninitialized_values);
    }

    GPU_LINE(__host__ __device__)
    static void modify_after_construction(Ray& main, const d3& starting_pos, const d3& ending_pos)
    {
        main.uninitialized_values = false;
        d3::assign(main.s, starting_pos);

        main.d.x = ending_pos.x - starting_pos.x;
        main.d.y = ending_pos.y - starting_pos.y;
        main.d.z = ending_pos.z - starting_pos.z;

        main.distance = d3::distance(main.d);

        main.d.x /= main.distance;
        main.d.y /= main.distance;
        main.d.z /= main.distance;
    }

    GPU_LINE(__host__ __device__)
    static void modify_after_construction_random_in_normal_hemisphere(Ray& main, const d3& starting_pos,
                                                                      const Ray& normal)
    {
        main.uninitialized_values = false;
        d3::assign(main.s, starting_pos);

        main.d.x = Random_functions::random_unit_minus_1_to_1();
        main.d.y = Random_functions::random_unit_minus_1_to_1();
        main.d.z = Random_functions::random_unit_minus_1_to_1();

        main.distance = d3::distance(main.d);

        main.d.normalize();

        if (Ray::dot_prod(main, normal) < 0)
        {
            main.d.negate();
        }
    }

    GPU_LINE(__host__ __device__)
    static void modify_after_constructionp_perfect_mirror(Ray& main, const d3& starting_pos,
                                                          const d3& hit_ray_direction, const Ray& normal)
    {
        main.uninitialized_values = false;
        d3::assign(main.s, starting_pos);

        d3 Incoming;
        d3 Normal;

        d3::assign(Incoming, hit_ray_direction);
        d3::assign(Normal, normal.d);

        unit dot_product = Incoming.dot_prod(Normal);
        main.d = Incoming - (Normal * (u(2) * dot_product));

        main.distance = d3::distance(main.d);
        main.d.normalize();
    }

    GPU_LINE(__host__)
    void print(const string& var_name) const
    {
        print_var_name;

        call_print(s);
        call_print(d);
        varr(distance);
        var(uninitialized_values);
    }

    GPU_LINE(__host__ __device__)
    void modify_after_construction(const d3& starting_pos, const d3& ending_pos)
    {
        uninitialized_values = false;

        s = starting_pos;

        ray_prep;
    }

    GPU_LINE(__host__ __device__)
    void modify_after_construction(const d3& starting_pos, const unit& d_x, const unit& d_y, const unit& d_z)
    {
        uninitialized_values = false;

        s = starting_pos;

        d.x = d_x;
        d.y = d_y;
        d.z = d_z;

        distance = d3::distance(d);
    }

    GPU_LINE(__host__ __device__)
    void modify_after_construction_random_in_normal_hemisphere(const d3& starting_pos, const Ray& normal)
    {
        uninitialized_values = false;

        s = starting_pos;

        d.x = Random_functions::random_unit_minus_1_to_1();
        d.y = Random_functions::random_unit_minus_1_to_1();
        d.z = Random_functions::random_unit_minus_1_to_1();

        distance = d3::distance(d);

        d.normalize();

        if (dot_prod(*this, normal) < 0)
        {
            d.negate();
        }
    }

    GPU_LINE(__host__ __device__)
    void modify_after_constructionp_perfect_mirror(const d3& starting_pos, const d3& hit_ray_direction,
                                                   const Ray& normal)
    {
        uninitialized_values = false;

        s = starting_pos;

        d3 Incoming = hit_ray_direction;
        d3 Normal = normal.d;

        unit dot_product = Incoming.dot_prod(Normal);
        d = Incoming - (Normal * (u(2) * dot_product));

        distance = d3::distance(d);
        d.normalize();
    }
};