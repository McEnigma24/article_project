#pragma once

struct Format_Buffer
{
    // do zapisywania logów
    static string input_variables_return_log_line(const tupl& variables)
    {
        char buffer[BUFFER_SIZE];

        sprintf(buffer,
                format

                    input_tupl_variables,
                get<1>(variables), get<2>(variables), get<3>(variables),
                get<0>(variables) // std_dev
        );

        return buffer;
    }

    // do czytania ich
    static tupl input_log_line_output_variables(const string& log_line)
    {
        u64 VALUE_min;
        u64 VALUE_avg;
        u64 VALUE_max;
        double standard_deviation;
        adding_buffer_init

        sscanf(log_line.c_str(),
               format

                   adding_buffer_use,
               &VALUE_min, &VALUE_avg, &VALUE_max, &standard_deviation);

        return {standard_deviation, VALUE_min, VALUE_avg, VALUE_max, adding_buffer_use};
    }
};