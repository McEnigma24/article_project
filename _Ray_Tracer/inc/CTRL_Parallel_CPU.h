#pragma once
#include "CTRL_STATS_Stat_Record.h"
#include "CTRL_Timer.h"
#include "_preprocessor_.h"
#include <barrier>

class Renderer;

class Parallel_CPU
{
public:
    enum Parallel_Schema
    {
        PR_solid_blocks,
        PR_subsequent,
        PR_subsequent_custom
    };

private:
    struct loop_info_per_thread
    {
        u64 start;
        u64 end;
        u64 inc;
        u64 block_size;

        loop_info_per_thread() { memset(this, 0, sizeof(*this)); }

        loop_info_per_thread(u64 s, u64 e, u64 i, u64 bs = 0) : start(s), end(e), inc(i), block_size(bs) {}

        GPU_LINE(__host__) void print(const string& name) const
        {
            nline;

            linee("loop_info_per_thread -> ");
            varr(name);
            varr(start);
            varr(end);
            varr(inc);
            varr(block_size);

            nline;
        }
    };

    // calling -> per pixel
    void thread_job_simple_schema(const loop_info_per_thread& info, int thread_id);
    void thread_job_complex_schema(const loop_info_per_thread& info, int thread_id);

public:
    // PREPARATIONS
    vector<loop_info_per_thread> prepare_subsets_for_threads_solid_blocks(u64 num_of_threads, u64 array_size);
    vector<loop_info_per_thread> prepare_subsets_for_threads_subsequent(u64 num_of_threads, u64 array_size);
    vector<loop_info_per_thread> prepare_subsets_for_threads_subsequent_blocks(u64 num_of_threads, u64 block_size,
                                                                               u64 array_size);

    Parallel_CPU(Renderer* _ptr);

    // MAIN FUNCTION
    void thread_group_host_and_round_keeper(int round_counter, const int num_of_threads, u16 parallel_schema,
                                            int custom_block_size, work_group& stats);

private:
    Renderer* ptr;
};