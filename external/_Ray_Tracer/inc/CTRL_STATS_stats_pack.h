#pragma once
#include "base/_preprocessor_.h"
#include <math.h>
#include <vector>

class stats_pack
{
    u64 min;
    u64 max;
    u64 sum;
    u64 count;
    vector<u64> data;

public:
    stats_pack()
    {
        data.reserve(G::REP_NUMBER);

        min = (0xFFFFFFFFFFFFFFFF);
        max = 0;
        sum = 0;
        count = 0;
    }

    void push(u64 p)
    {
        if (p < min) min = p;
        if (max < p) max = p;

        sum += p;
        count++;

        data.push_back(p);
    }

    double calculate_relative_standard_deviation()
    {
        double mean = ((double)sum) / ((double)count);
        double sum_of_all_dev_square = 0.0;

        for (auto& x : data)
        {
            sum_of_all_dev_square += pow2(x - mean);
        }
        sum_of_all_dev_square /= (double)count; // population

        double standard_deviation = sqrt(sum_of_all_dev_square);

        return 100 * (standard_deviation / mean);
    }

    tuple<u64, u64, u64, double> get()
    {
        return {min, static_cast<u64>(((double)sum) / ((double)count)), max, calculate_relative_standard_deviation()};
    }

    vector<u64>& get_data() { return data; }
};
