#pragma once

// COMMON //

#define BUFFER_SIZE 10000

#define add(a, x, b) a##x##b
#define add_array(x) char buffer_##x[BUFFER_SIZE]

#define adding_read                                                                                                    \
    add(read_, VALUE_rel_dev, ), add(read_, VALUE_min, ), add(read_, VALUE_avg, ), add(read_, VALUE_max, ),            \
        add_to_all(read_, )
#define adding_buffer_use add_to_all(buffer_, )

void insert_raw_datapoints(string& output, const vector<u64> data)
{
    output += " {";
    for (auto& data_point : data)
    {
        output += " " + std::to_string(data_point);
    }
    output += " }";
}