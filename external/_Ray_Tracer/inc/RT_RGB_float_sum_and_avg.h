#pragma once
#include "RT_RGB_float.h"
#include "base/_preprocessor_.h"

class RGB_float_sum_and_avg
{
public:
    u64 counter;
    unit r;
    unit g;
    unit b;

    GPU_LINE(__host__ __device__)
    RGB_float_sum_and_avg() : counter(0), r(u(0)), g(u(0)), b(u(0)) {}

    GPU_LINE(__host__ __device__)
    static void init(RGB_float_sum_and_avg& obj)
    {
        obj.counter = 0;
        obj.r = 0;
        obj.g = 0;
        obj.b = 0;
    }

    GPU_LINE(__host__ __device__)
    static void assign(RGB_float_sum_and_avg& main, const RGB_float_sum_and_avg& other)
    {
        main.counter = other.counter;
        main.r = other.r;
        main.g = other.g;
        main.b = other.b;
    }

    GPU_LINE(__host__ __device__)
    static void add_color(RGB_float_sum_and_avg& main, const RGB_float& adding)
    {
        main.r += adding.r;
        main.g += adding.g;
        main.b += adding.b;

        main.counter++;
    }

    GPU_LINE(__host__ __device__)
    static void make_average(RGB_float_sum_and_avg& main)
    {
        if (main.counter <= 1) return;

        main.r /= main.counter;
        main.g /= main.counter;
        main.b /= main.counter;
    }

    GPU_LINE(__host__ __device__)
    static void return_RGB_float(RGB_float_sum_and_avg& main, RGB_float& ret)
    {
        RGB_float_sum_and_avg::make_average(main);

        ret.r = main.r;
        ret.g = main.g;
        ret.b = main.b;
    }

    void print(const string& var_name) const
    {
        print_var_name;

        varr(counter);
        varr(r);
        varr(g);
        var(b);
    }

    GPU_LINE(__host__ __device__)
    u64 check_counter() const { return counter; }

    GPU_LINE(__host__ __device__)
    void add_color(const RGB_float& adding_color)
    {
        auto [add_r, add_g, add_b] = adding_color.get();

        r += add_r;
        g += add_g;
        b += add_b;

        counter++;
    }

    GPU_LINE(__host__ __device__)
    RGB_float_sum_and_avg& operator+=(const RGB_float& other)
    {
        this->add_color(other);
        return *this;
    }

    GPU_LINE(__host__ __device__)
    void make_average()
    {
        if (counter <= 1) return;

        r /= counter;
        g /= counter;
        b /= counter;
    }

    GPU_LINE(__host__ __device__)
    RGB_float_sum_and_avg& operator=(const RGB_float_sum_and_avg& other)
    {
        THIS_OTHER(counter);
        THIS_OTHER(r);
        THIS_OTHER(g);
        THIS_OTHER(b);

        return *this;
    }

#ifdef WIN
    GPU_LINE(__host__ __device__)
    Color return_Color()
    {
        make_average();

        return Color(static_cast<sf::Uint8>(r * 255), static_cast<sf::Uint8>(g * 255), static_cast<sf::Uint8>(b * 255));
    }
#endif // WIN

    GPU_LINE(__host__ __device__)
    RGB_float return_RGB_float()
    {
        make_average();

        return {r, g, b};
    }
};
