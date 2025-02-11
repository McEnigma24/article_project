#!/bin/bash

DIR_ROOT=$(dirname "$(pwd)")
DIR_BUILD="build"
DIR_LOG="log"
DIR_TARGET="exe"
DIR_OUTPUT="output"

clear_dir() { if [ -d $1 ]; then rm -rf $1; fi; mkdir $1; }
run_and_collect()
{
    total_files=$(ls -1 $DIR_TARGET/* | wc -l)
    current_file=1

    for exe in $DIR_TARGET/*; do
    {
        log_name=$(basename $exe); log_name="${log_name%.*}";
        
        # echo -e "\nRUN ($current_file/$total_files) - $exe"; ./$exe > $DIR_LOG/$log_name.log;
        echo -e "\nRUN ($current_file/$total_files) - $exe"; ./$exe

        current_file=$((current_file + 1))
    }
    done
}


# START #

./production.sh; echo -e "\n"

cd $DIR_ROOT
clear_dir "$DIR_LOG"
clear_dir "$DIR_OUTPUT"

run_and_collect