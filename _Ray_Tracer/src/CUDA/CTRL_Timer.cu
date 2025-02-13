#include "CTRL_Timer.h"



string Timer::how_many_spaces_before(const u64& number)
{
    if (number < 10) return "  ";
    else if (number < 100) return " ";
    return "";
}
tuple<u64, u64, u64, u64, u64> Timer::convert_nano_to_all_units_remainer(const u64& all_in_nano_seconds)
{
    u64 minutes = static_cast<u64>(static_cast<double>(all_in_nano_seconds) / static_cast<double>(60000000000));
    u64 seconds = static_cast<u64>(static_cast<double>(all_in_nano_seconds - (minutes * 60000000000)) / static_cast<double>(1000000000));
    u64 mili = static_cast<u64>(static_cast<double>(all_in_nano_seconds - (minutes * 60000000000) - (seconds * 1000000000)) / static_cast<double>(1000000));
    u64 micro = static_cast<u64>(static_cast<double>(all_in_nano_seconds - (minutes * 60000000000) - (seconds * 1000000000) - (mili * 1000000)) / static_cast<double>(1000));
    u64 nano = static_cast<u64>(all_in_nano_seconds - (minutes * 60000000000) - (seconds * 1000000000) - (mili * 1000000) - (micro * 1000));


    return {
                minutes,
                seconds,
                mili,
                micro,
                nano
    };
}
tuple<double, double, double, double, double> Timer::convert_nano_to_all_units_just_devisions(const u64& all_in_nano_seconds)
{
    double minutes = static_cast<double>(all_in_nano_seconds) / (1e9 * 60);
    double seconds = static_cast<double>(all_in_nano_seconds) / (1e9);
    double mili = static_cast<double>(all_in_nano_seconds) / (1e6);
    double micro = static_cast<double>(all_in_nano_seconds) / (1e3);
    double nano = static_cast<double>(all_in_nano_seconds);


    return {
                minutes,
                seconds,
                mili,
                micro,
                nano
    };
}
tuple<u64, u64, u64, u64, u64> Timer::remaining_time()
{
    u64 all_in_nano_seconds = get_all_in_nano();

    return convert_nano_to_all_units_remainer(all_in_nano_seconds);
}


void Timer::log_cout(tuple<u64, u64, u64, u64, u64> all, u64 all_nano)
{
    auto [minutes, seconds, mili, micro, nano] = all;
    cout << minutes << "m " << seconds << "s " << mili << "mil " << micro << "mic " << nano << "n  -  " << all_nano << '\n';
}


void Timer::log()
{
    log_cout(remaining_time(), get_all_in_nano());
}
void Timer::log(u64 all_in_nano_seconds)
{
    log_cout(convert_nano_to_all_units_remainer(all_in_nano_seconds), all_in_nano_seconds);
}

Timer::Timer()
{
}

u64 Timer::get_all_in_nano()
{
    return std::chrono::duration_cast<chrono::nanoseconds>(m_end - m_start).count();
}

void Timer::start()
{
    m_start = std::chrono::high_resolution_clock::now();
}
void Timer::end()
{
    m_end = std::chrono::high_resolution_clock::now();
}