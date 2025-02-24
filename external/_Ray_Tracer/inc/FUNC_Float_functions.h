#pragma once
#include "base/_preprocessor_.h"

struct Floating_functions
{
    GPU_LINE(__host__ __device__)
    inline static bool comparison_equal_to_zero(unit val) { return (abs(val) < def_EPSILON); }

    GPU_LINE(__host__ __device__)
    inline static bool comparison_bigger_than_zero(unit val) { return (def_EPSILON < val); }

    GPU_LINE(__host__ __device__)
    inline static bool comparison_smaller_than_zero(unit val) { return (val < -def_EPSILON); }

    GPU_LINE(__host__ __device__)
    inline static bool comparison_equal(unit a, unit b) { return comparison_equal_to_zero(a - b); }

    GPU_LINE(__host__ __device__)
    inline static bool comparison_bigger___true_if_a_bigger_than_b(unit a, unit b) { return comparison_bigger_than_zero(a - b); }

    GPU_LINE(__host__ __device__)
    inline static bool comparison_smaller___true_if_a_smaller_than_b(unit a, unit b) { return comparison_smaller_than_zero(a - b); }
};
