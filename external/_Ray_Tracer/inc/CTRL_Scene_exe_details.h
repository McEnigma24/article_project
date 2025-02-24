#pragma once
#include "base/_preprocessor_.h"

struct exe_details
{
    u16 num_of_threads;
    u16 schema;
    u16 block_size;

    exe_details(u16 _num_of_threads, u16 _schema, u16 _block_size) : c_init(num_of_threads), c_init(schema), c_init(block_size) {}
};