#pragma once
#include "base/_type_definitions_.h"
#include <cstdlib>
#include <ctime>
#include <random>

static int div_range = 10000;

struct Random_functions
{
    static unit random_unit_0_to_1()
    {
        // not used, every run make the same 'random' numbers

        /*static std::random_device rd;
        static std::mt19937 e2(rd());
        static std::uniform_real_distribution<> dist(0, 1);
        return dist(e2);*/

        return u(rand() % div_range + 1) / div_range;
    }

    static unit random_unit_minus_1_to_1()
    {
        // not used, every run make the same 'random' numbers

        /*static std::random_device rd;
        static std::mt19937 e2(rd());
        static std::uniform_real_distribution<> dist(-1, 1);
        return dist(e2);*/

        return (u(rand() % div_range + 1) / (div_range / 2)) - u(1);
    }

    static int distance_between_numbers(int a, int b)
    {
        int ret = a - b;
        return (ret > 0) ? ret : -ret;
    }

    static int random_int_in_range(int a, int b)
    {
        int range = distance_between_numbers(a, b);

        int r = rand() % range;

        int smaller = (a < b) ? a : b;

        return r + smaller;
    }
};