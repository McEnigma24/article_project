#pragma once
#include "CTRL_STATS_stats_pack.h"
#include "base/_preprocessor_.h"

class work_group
{
    u16 how_many_threads_in_group;
    stats_pack whole_group_time_report;

#ifdef def_STATS_PER_THREAD
    vector<stats_pack> single_thread;
#endif

public:
    work_group(int _how_many_threads_in_group)
        : how_many_threads_in_group(_how_many_threads_in_group)
#ifdef def_STATS_PER_THREAD
          ,
          single_thread(_how_many_threads_in_group)
#endif
    {
    }

    void push_whole(const u64& p) { whole_group_time_report.push(p); }

    tuple<u64, u64, u64, double> get_whole() { return whole_group_time_report.get(); }

    vector<u64>& get_whole_data() { return whole_group_time_report.get_data(); }

#ifdef def_STATS_PER_THREAD
    void push_single(const u64& p, u64 i) { single_thread[i].push(p); }

    tuple<u64, u64, u64, double> get_single(u64 i) { return single_thread[i].get(); }
#endif
};
