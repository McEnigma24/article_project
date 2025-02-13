#pragma once
#include "_preprocessor_.h"
#include <random>

class RGB
{
public:
    u8 r;
    u8 g;
    u8 b;

#ifdef WIN
    GPU_LINE(__host__ __device__)
    RGB(const Color& color) : r(color.r), g(color.g), b(color.b) {}
    //   0%         100%
    GPU_LINE(__host__ __device__)
    Color gradient_between_0_to_1(RGB col_1, RGB col_2, unit percent)
    {
        RGB ret = {col_1};

        col_1.scale(percent);
        col_2.scale(percent);

        ret -= col_1;
        ret += col_2;

        return ret;
    }
    GPU_LINE(__host__ __device__)
    Color gradient_between_neg_1_to_1(RGB col_1, RGB col_2, unit percent)
    {
        return gradient_between_0_to_1(col_1, col_2, (percent + 1) / 2);
    }
    GPU_LINE(__host__ __device__)
    operator Color() { return Color(r, g, b); }
    GPU_LINE(__host__ __device__)
    Color ret_Color() { return Color(r, g, b); }
#endif // WIN

    GPU_LINE(__host__ __device__)
    RGB(u8 p_r = 0, u8 p_g = 0, u8 p_b = 0) : r(p_r), g(p_g), b(p_b) {}

    GPU_LINE(__host__ __device__)
    RGB(const RGB& other) : r(other.r), g(other.g), b(other.b) {}

    GPU_LINE(__host__)
    void print(const string& var_name) const
    {
        print_var_name;

        varr((int)r);
        varr((int)g);
        var((int)b);
    }

    GPU_LINE(__host__ __device__)
    u8 get_r() const { return r; }
    GPU_LINE(__host__ __device__)
    u8 get_g() const { return g; }
    GPU_LINE(__host__ __device__)
    u8 get_b() const { return b; }

    GPU_LINE(__host__ __device__)
    void scale(unit percent)
    {
        r = static_cast<u8>(r * percent);
        g = static_cast<u8>(r * percent);
        b = static_cast<u8>(r * percent);
    }

    GPU_LINE(__host__ __device__)
    RGB& operator+=(const RGB& other)
    {
        r += other.r;
        g += other.g;
        b += other.b;

        return *this;
    }
    GPU_LINE(__host__ __device__)
    RGB operator+(const RGB& other) { return RGB(r + other.r, g + other.g, b + other.b); }
    GPU_LINE(__host__ __device__)
    RGB& operator-=(const RGB& other)
    {
        r -= other.r;
        g -= other.g;
        b -= other.b;

        return *this;
    }
    GPU_LINE(__host__ __device__)
    RGB operator-(const RGB& other) { return RGB(r - other.r, g - other.g, b - other.b); }

    GPU_LINE(__host__ __device__)
    RGB operator*(unit multi)
    {
        return {static_cast<u8>(r * multi), static_cast<u8>(g * multi), static_cast<u8>(b * multi)};
    }
};
