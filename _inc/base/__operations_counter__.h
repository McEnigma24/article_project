#pragma once
#include "__preprocessor__.h"
#include <atomic>

#ifdef OPERATION_COUNTER

#ifdef OPERATION_COUNTER_SHOW_LOG
#define OP_C_SHOW_LOG_LINE(...) __VA_ARGS__
#else
#define OP_C_SHOW_LOG_LINE(...)
#endif // OPERATION_COUNTER_SHOW_LOG

struct Global_Operation_Counter
{
    class Counters
    {
        atomic<uint64_t> counter_add;
        atomic<uint64_t> counter_sub;
        atomic<uint64_t> counter_multi;
        atomic<uint64_t> counter_dev;
        atomic<uint64_t> counter_mod;
        atomic<uint64_t> counter_comparisons;

    public:
        Counters() { memset(this, 0, sizeof(*this)); }
        void show()
        {
            var(counter_add);
            var(counter_sub);
            var(counter_multi);
            var(counter_dev);
            var(counter_mod);
            var(counter_comparisons);
        }

        void add() { counter_add++; }
        void sub() { counter_sub++; }
        void multi() { counter_multi++; }
        void dev() { counter_dev++; }
        void mod() { counter_mod++; }
        void comparison() { counter_comparisons++; }
    };
    static Counters counters;

    static void reset() { memset(&counters, 0, sizeof(counters)); }

    static void show()
    {
        nline;
        line("< Operation Counters >");
        counters.show();
        nline;
    }
};

template <typename T>
class Operation_Counter
{
    T value;

public:
    Operation_Counter() { memset(this, 0, sizeof(*this)); }
    Operation_Counter(const T& v) : value(v)
    {
        OP_C_SHOW_LOG_LINE(line("constructor"));
    }
    Operation_Counter(Operation_Counter&& v) : value(v.value)
    {
        OP_C_SHOW_LOG_LINE(line("move constructor"));
    }
    Operation_Counter(const Operation_Counter& other) : value(other.value)
    {
        OP_C_SHOW_LOG_LINE(line("copy constructor"));
    }
    Operation_Counter& operator=(const T& other_value)
    {
        OP_C_SHOW_LOG_LINE(line("operator="));
        value = other_value;
        return *this;
    }
    Operation_Counter& operator=(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator="));
        value = other.value;
        return *this;
    }

    bool operator==(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator=="));

        Global_Operation_Counter::counters.comparison();
        return value == other.value;
    }
    bool operator!=(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator!="));

        Global_Operation_Counter::counters.comparison();
        return value != other.value;
    }
    bool operator>(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator>"));

        Global_Operation_Counter::counters.comparison();
        return value > other.value;
    }
    bool operator<(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator<"));

        Global_Operation_Counter::counters.comparison();
        return value < other.value;
    }
    bool operator>=(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator>="));

        Global_Operation_Counter::counters.comparison();
        return value >= other.value;
    }
    bool operator<=(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator<="));

        Global_Operation_Counter::counters.comparison();
        return value <= other.value;
    }

    Operation_Counter operator+(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator+"));

        Global_Operation_Counter::counters.add();
        return value + other.value;
    }
    Operation_Counter& operator+=(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator+="));

        Global_Operation_Counter::counters.add();
        value += other.value;
        return *this;
    }
    Operation_Counter operator+(const T& other_value)
    {
        OP_C_SHOW_LOG_LINE(line("operator+"));

        Global_Operation_Counter::counters.add();
        return value + other_value;
    }
    Operation_Counter& operator+=(const T& other_value)
    {
        OP_C_SHOW_LOG_LINE(line("operator+="));

        Global_Operation_Counter::counters.add();
        value += other_value;
        return *this;
    }
    Operation_Counter& operator++()
    {
        OP_C_SHOW_LOG_LINE(line("operator++ pre"));

        Global_Operation_Counter::counters.add();
        value++;
        return *this;
    }
    Operation_Counter operator++(int)
    {
        OP_C_SHOW_LOG_LINE(line("operator++ post"));

        Global_Operation_Counter::counters.add();
        Operation_Counter previous_value = *this;
        value++;
        return previous_value;
    }

    Operation_Counter operator-(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator-"));

        Global_Operation_Counter::counters.sub();
        return value - other.value;
    }
    Operation_Counter& operator-=(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator-="));

        Global_Operation_Counter::counters.sub();
        value -= other.value;
        return *this;
    }
    Operation_Counter operator-(const T& other_value)
    {
        OP_C_SHOW_LOG_LINE(line("operator-"));

        Global_Operation_Counter::counters.sub();
        return value - other_value;
    }
    Operation_Counter& operator-=(const T& other_value)
    {
        OP_C_SHOW_LOG_LINE(line("operator-="));

        Global_Operation_Counter::counters.sub();
        value -= other_value;
        return *this;
    }
    Operation_Counter& operator--()
    {
        OP_C_SHOW_LOG_LINE(line("operator-- pre"));

        Global_Operation_Counter::counters.sub();
        value--;
        return *this;
    }
    Operation_Counter operator--(int)
    {
        OP_C_SHOW_LOG_LINE(line("operator-- post"));

        Global_Operation_Counter::counters.sub();
        Operation_Counter previous_value = *this;
        value--;
        return previous_value;
    }

    Operation_Counter operator*(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator*"));

        Global_Operation_Counter::counters.multi();
        return value * other.value;
    }
    Operation_Counter& operator*=(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator*="));

        Global_Operation_Counter::counters.multi();
        value *= other.value;
        return *this;
    }
    Operation_Counter operator*(const T& other_value)
    {
        OP_C_SHOW_LOG_LINE(line("operator*"));

        Global_Operation_Counter::counters.multi();
        return value * other_value;
    }
    Operation_Counter& operator*=(const T& other_value)
    {
        OP_C_SHOW_LOG_LINE(line("operator*="));

        Global_Operation_Counter::counters.multi();
        value *= other_value;
        return *this;
    }

    Operation_Counter operator/(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator/"));

        Global_Operation_Counter::counters.dev();
        return value / other.value;
    }
    Operation_Counter& operator/=(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator/="));

        Global_Operation_Counter::counters.dev();
        value /= other.value;
        return *this;
    }
    Operation_Counter operator/(const T& other_value)
    {
        OP_C_SHOW_LOG_LINE(line("operator/"));

        Global_Operation_Counter::counters.dev();
        return value / other_value;
    }
    Operation_Counter& operator/=(const T& other_value)
    {
        OP_C_SHOW_LOG_LINE(line("operator/="));

        Global_Operation_Counter::counters.dev();
        value /= other_value;
        return *this;
    }

    Operation_Counter operator%(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator%"));

        Global_Operation_Counter::counters.mod();
        return value % other.value;
    }
    Operation_Counter& operator%=(const Operation_Counter& other)
    {
        OP_C_SHOW_LOG_LINE(line("operator%="));

        Global_Operation_Counter::counters.mod();
        value %= other.value;
        return *this;
    }
    Operation_Counter operator%(const T& other_value)
    {
        OP_C_SHOW_LOG_LINE(line("operator%"));

        Global_Operation_Counter::counters.mod();
        return value % other_value;
    }
    Operation_Counter& operator%=(const T& other_value)
    {
        OP_C_SHOW_LOG_LINE(line("operator%="));

        Global_Operation_Counter::counters.mod();
        value %= other_value;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& stream,
                                    const Operation_Counter& obj)
    {
        // line("stream"));
        stream << obj.value;
        return stream;
    }
};

// comment out what you don't want to count //

#define i8 Operation_Counter<i8>
#define i16 Operation_Counter<i16>
#define i32 Operation_Counter<i32>
#define i64 Operation_Counter<i64>

#define u8 Operation_Counter<u8>
#define u16 Operation_Counter<u16>
#define u32 Operation_Counter<u32>
#define u64 Operation_Counter<u64>

#define float Operation_Counter<float>
#define double Operation_Counter<double>

#endif // OPERATION_COUNTER