#!/bin/bash

SCALING_FROM_START_multi="$1"
SCALING_FROM_START_add="$2"
DIR_ROOT=$(dirname "$(pwd)")
DIR_BUILD="build"
DIR_LOG="log"
DIR_TARGET="exe"
DIR_OUTPUT="output"
path_concatenated_log="combined_data.log"

clear_dir() { if [ -d $1 ]; then rm -rf $1; fi; mkdir $1; }
create_dir() { if [ ! -d $1 ]; then mkdir $1; fi; }
run_and_collect()
{
    total_files=$(ls -1 $DIR_TARGET/* | wc -l)
    current_file=1

    cpu_n=$(lscpu | grep "Model name" | awk -F: '{print $2}' | sed 's/^ *//')
    cpu_name=$(echo $cpu_n | tr '[:space:]-()@.' '_')

    gpu_n=$(nvidia-smi -L | awk -F': ' '{print $2}' | awk -F'(' '{print $1}')
    gpu_name=$(echo $gpu_n | tr '[:space:]-()@.' '_')

    # gpu_n=$(glxinfo | grep "Device" | awk -F'[()]' '{print $2}')
    # gpu_name=$(echo $gpu_n | tr '[:space:]-()@.' '_')

    physical_cores=$(lscpu | grep "^Core(s) per socket:" | awk '{print $4}')

    if echo "$gpu_name" | grep -iq "nvidia"; then

        echo "Nvidia GPU found"

    else

        echo "No Nvidia GPU found"
        gpu_name="nvidia_gpu_not_present"

    fi

    # cpu_name="fake_CPU"
    # gpu_name="fake_GPU"

    ONE="1"
    current_add=0

    for exe in $DIR_TARGET/*; do
    {
        log_name=$(basename $exe); log_name="${log_name%.*}";

        if [ "$SCALING_FROM_START_add" != "" ]; then
            scaling_multi=$(echo "scale=50; $ONE / $total_files" | bc)
            scaling_add=$(echo "scale=50; $scaling_multi * $current_add" | bc)

            scaling_multi=$(echo "scale=50; $scaling_multi * $SCALING_FROM_START_multi" | bc)
            scaling_add=$(echo "scale=50; $scaling_add * $SCALING_FROM_START_multi" | bc)

            scaling_add=$(echo "scale=50; $scaling_add + $SCALING_FROM_START_add" | bc)

            echo -e "\nRUN ($current_file/$total_files) - $exe"; ./$exe "$cpu_name" "$gpu_name" "$physical_cores" "$scaling_multi" "$scaling_add"
            if [ $? -eq 0 ]; then
                echo -n ""
            else
                echo -e "\nrun.sh - ERROR - error when running $exe"
                exit
            fi
        else
            echo -e "\nRUN ($current_file/$total_files) - $exe"; ./$exe "$cpu_name" "$gpu_name" "$physical_cores" > $DIR_LOG/$log_name.log;
        fi

        current_file=$((current_file + 1))
        current_add=$((current_add + 1))
    }
    done

    # pakowanie

    echo -e "\nBenchmark Completed\n"

    echo "Packing logs"
    cd $DIR_OUTPUT

    is_first=true
    for file in *.txt; do
        if [[ -f "$file" ]]; then
            if [[ "$is_first" == true ]]; then
                cat $file > $path_concatenated_log
                is_first=false
            else
                cat $file >> $path_concatenated_log
            fi
            rm -f $file
        fi
    done

    data=$(date +"%Y_%m_%d_%H_%M_%S")
    find . -maxdepth 1 -type f -print0 | tar --null -cf $data.tar --files-from=-

    echo -e "\nMain Ray Tracer is DONE"
}


# START #

./production.sh "log_to_terminal"; echo -e "\n"

cd $DIR_ROOT
clear_dir "$DIR_LOG"
clear_dir "$DIR_OUTPUT"
create_dir "$DIR_OUTPUT/img"

run_and_collect