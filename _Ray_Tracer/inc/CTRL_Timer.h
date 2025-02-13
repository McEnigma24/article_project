#pragma once
#include "_preprocessor_.h"

class Timer
{
    chrono_clock m_start;
    chrono_clock m_end;

private:
    string how_many_spaces_before(const u64& number);
    static tuple<u64, u64, u64, u64, u64> convert_nano_to_all_units_remainer(const u64& all_in_nano_seconds);
    static tuple<double, double, double, double, double>
    convert_nano_to_all_units_just_devisions(const u64& all_in_nano_seconds);
    tuple<u64, u64, u64, u64, u64> remaining_time();

    static void log_cout(tuple<u64, u64, u64, u64, u64> all, u64 all_nano);

public:
    void log();
    static void log(u64 all_in_nano_seconds);

    Timer();

    u64 get_all_in_nano();

    void start();
    void end();
};