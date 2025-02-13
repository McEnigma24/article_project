#pragma once
#include "_pre_code_mode_.h"

#ifdef WIN
#include <SFML/Graphics.hpp>
#endif

#include <chrono>

WIN_LINE(using namespace sf;)
using namespace std;

typedef std::chrono::high_resolution_clock::time_point chrono_clock;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#define M_PI 3.14159265358979323846

#ifdef UNIT_FLOAT

typedef float unit;

#endif

#ifdef UNIT_DOUBLE

typedef double unit;

#endif

#define u(x) (static_cast<unit>(x))

enum Surface_type
{
    diffuse,
    mirror,
    pass_through
};