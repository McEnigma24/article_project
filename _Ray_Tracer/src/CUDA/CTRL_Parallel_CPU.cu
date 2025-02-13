#include "CTRL_Parallel_CPU.h"
#include "RT_Renderer.h"

#ifdef WIN
#define num_of_colors 5
Color color_tab[num_of_colors] =
{
    Color(197,64,64),
    Color(197,141,54),
    Color(101,195,54),
    Color(50,193,190),
    Color(72,61,190)
};
#endif // WIN

#ifdef CPU

// PREPARATIONS
vector<Parallel_CPU::loop_info_per_thread> Parallel_CPU::prepare_subsets_for_threads_solid_blocks(u64 num_of_threads, u64 array_size)
{
    vector<loop_info_per_thread> ret;
    u64 inc = 1;

    u64 block_size = array_size / num_of_threads;
    u64 i;
    for (i = 0; i < (num_of_threads - 1); i++)
    {
        u64 start = i * block_size;
        u64 end = (i + 1) * block_size - 1;
        ret.emplace_back(start, end, inc);
    }

    // LAST SET
    u64 start = i * block_size;
    u64 end = array_size - 1;
    ret.emplace_back(start, end, inc);

    return ret;
}
vector<Parallel_CPU::loop_info_per_thread> Parallel_CPU::prepare_subsets_for_threads_subsequent(u64 num_of_threads, u64 array_size)
{
    vector<loop_info_per_thread> ret;
    u64 end = array_size - 1;
    u64 inc = num_of_threads;

    for (u64 i = 0; i < num_of_threads; i++)
    {
        u64 start = i;
        ret.emplace_back(start, end, inc);
    }

    return ret;
}
vector<Parallel_CPU::loop_info_per_thread> Parallel_CPU::prepare_subsets_for_threads_subsequent_blocks(u64 num_of_threads, u64 block_size, u64 array_size)
{
    vector<loop_info_per_thread> ret;
    u64 end = array_size - 1;
    u64 inc = num_of_threads * block_size;

    for (u64 i = 0; i < num_of_threads; i++)
    {
        u64 start = i * block_size;
        ret.emplace_back(start, end, inc, block_size);
    }

    return ret;
}

// calling -> per pixel

#define func_call(x) ptr->per_pixel(x, G::Render::current_scene, ptr->get_my_pixel(), G::Render::current_scene->get_lights_ptr(), G::Render::current_scene->get_spheres_ptr(), G::Render::current_scene->get_details_ptr());
//#define func_call(x) ptr->set_pixel_hack_for_showing_schema(x, color_tab[thread_id % num_of_colors]);

void Parallel_CPU::thread_job_simple_schema(const Parallel_CPU::loop_info_per_thread& info, int thread_id)
{
    for (u64 i = info.start; i <= info.end; i += info.inc)
    {
        func_call(i);
    }
}
void Parallel_CPU::thread_job_complex_schema(const Parallel_CPU::loop_info_per_thread& info, int thread_id)
{
    for (u64 i = info.start; i <= info.end; i += info.inc)
    {
        if (info.end <= i + info.block_size)
        {
            for (u64 j = i; j <= info.end; j++)
            {
                func_call(j);
            }
            return;
        }

        for (u64 j = i; j < i + info.block_size; j++)
        {
            func_call(j);
        }
    }
}



Parallel_CPU::Parallel_CPU(Renderer* _ptr)
    :c_init(ptr)
{
}

// MAIN FUNCTION
void Parallel_CPU::thread_group_host_and_round_keeper(int round_counter, const int num_of_threads, u16 parallel_schema, int custom_block_size, work_group& stats)
{
    round_counter++;
    const int all_rounds = round_counter;
    u64 array_size = G::WIDTH * G::HEIGHT;

    Timer t_whole;

    // SYNC function executed by one thread
    auto on_completion = [&]() noexcept
    {
        // skip first one, nothing to stop
        if (all_rounds != round_counter)
        {
            t_whole.end();
            stats.push_whole(t_whole.get_all_in_nano());
        }

        // NEXT ROUND
        round_counter--;

        // don't start on the last one
        if (round_counter != 0)
        {
            t_whole.start();
        }
    };

    // SYNC
    std::barrier sync_point(num_of_threads, on_completion);

    // Per thread timer or just group
    #ifdef def_STATS_PER_THREAD
    #define work(x) \
    Timer t; \
    for (; round_counter != 1;) \
    { \
        sync_point.arrive_and_wait(); \
        t.start(); \
        x(info, thread_id); \
        t.end(); \
        stats.push_single(t.get_all_in_nano(), thread_id); \
    } \
    sync_point.arrive_and_wait();
    #else
    #define work(x) \
    for (; round_counter != 1;) \
    { \
        sync_point.arrive_and_wait(); \
        x(info, thread_id); \
    } \
    sync_point.arrive_and_wait();
    #endif

    // creating functions for both schemas
    auto work_comp = [&](int thread_id, const loop_info_per_thread& info)
    {
        work(thread_job_complex_schema);
    };
    auto work_simple = [&](int thread_id, const loop_info_per_thread& info)
    {
        work(thread_job_simple_schema);
    };

    vector<jthread> threads;
    threads.reserve(num_of_threads);

    // starting threads with correct work call by schema
    vector<loop_info_per_thread> schema;
    if (parallel_schema == PR_solid_blocks || parallel_schema == PR_subsequent)
    {
        if (parallel_schema == PR_solid_blocks) schema = prepare_subsets_for_threads_solid_blocks(num_of_threads, array_size);
        else schema = prepare_subsets_for_threads_subsequent(num_of_threads, array_size);
        

        for (int i = 0; i < num_of_threads; i++) { threads.emplace_back(work_simple, i, schema[i]); }
    }
    else
    {
        schema = prepare_subsets_for_threads_subsequent_blocks(num_of_threads, custom_block_size, array_size);

        for (int i = 0; i < num_of_threads; i++) { threads.emplace_back(work_comp, i, schema[i]); }
    }
}

#endif
