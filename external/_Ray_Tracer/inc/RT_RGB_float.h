#pragma once
#include "RT_RGB.h"
#include "_preprocessor_.h"

class RGB_float
{
public:
    unit r;
    unit g;
    unit b;

    GPU_LINE(__host__ __device__)
    RGB_float(const RGB normal_color)
    {
        r = static_cast<unit>(normal_color.get_r()) / 255;
        g = static_cast<unit>(normal_color.get_g()) / 255;
        b = static_cast<unit>(normal_color.get_b()) / 255;
    }
    GPU_LINE(__host__ __device__)
    RGB_float(unit _r = u(0), unit _g = u(0), unit _b = u(0)) : r(_r), g(_g), b(_b) {}

    GPU_LINE(__host__ __device__)
    static void assign(RGB_float& main, const RGB_float& other)
    {
        main.r = other.r;
        main.g = other.g;
        main.b = other.b;
    }

    GPU_LINE(__host__ __device__)
    static void mix_with_unit(RGB_float& main, unit multi)
    {
        main.r *= multi;
        main.g *= multi;
        main.b *= multi;
    }

    GPU_LINE(__host__ __device__)
    static void mix_with_RGB_float(RGB_float& main, RGB_float& other)
    {
        main.r *= other.r;
        main.g *= other.g;
        main.b *= other.b;
    }

    void print(const string& var_name) const
    {
        print_var_name;

        varr(r);
        varr(g);
        var(b);
    }

    GPU_LINE(__host__ __device__)
    tuple<unit, unit, unit> get() const { return {r, g, b}; }

    GPU_LINE(__host__ __device__)
    RGB_float mix_with_RGB(const RGB color) const { return {color.get_r() * r, color.get_g() * g, color.get_b() * b}; }
    GPU_LINE(__host__ __device__)
    RGB_float mix_with_RGB_float(const RGB_float& color) const { return {color.r * r, color.g * g, color.b * b}; }

    GPU_LINE(__host__ __device__)
    operator RGB() { return {static_cast<u8>(r * 255), static_cast<u8>(g * 255), static_cast<u8>(b * 255)}; }

#ifdef WIN
    GPU_LINE(__host__ __device__)
    operator Color() { return {static_cast<Uint8>(r * 255), static_cast<Uint8>(g * 255), static_cast<Uint8>(b * 255)}; }
#endif // WIN

    GPU_LINE(__host__ __device__)
    RGB_float operator*(unit multi) { return {(r * multi), (g * multi), (b * multi)}; }
};
