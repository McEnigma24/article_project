#pragma once
#include "RT_RGB.h"
#include "_preprocessor_.h"

struct BMP_static
{
    static void save(const string& file_name, const vector<RGB>& my_pixel);
};