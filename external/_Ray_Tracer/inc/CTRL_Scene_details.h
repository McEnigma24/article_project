#pragma once
#include "base/_preprocessor_.h"

struct details
{
    u16 for_lights;
    u16 for_spheres;
    u16 for_bounces;

    details(u16 l, u16 s, u16 b) : for_lights(l), for_spheres(s), for_bounces(b) {}
};