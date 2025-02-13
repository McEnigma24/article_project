#!/bin/bash

SCALING_FOR_START_multi="$1"
SCALING_FOR_START_add="$2"
var_start=""

timer_start()
{
    var_start=$(date +%s)
}
timer_end()
{
    var_end=$(date +%s)
    elapsed=$((var_end - var_start))
    hours=$((elapsed / 3600))
    minutes=$(( (elapsed % 3600) / 60 ))
    seconds=$((elapsed % 60))
    printf "Ray Tracer - took: %02d:%02d:%02d\n" $hours $minutes $seconds
}

chmod +x scripts/*.sh

SCRIPT="./run.sh"
LOG="../start.log"


cd scripts
timer_start
if [ "$SCALING_FOR_START_add" != "" ]; then
    $SCRIPT "$SCALING_FOR_START_multi" "$SCALING_FOR_START_add"
else
    clear
    # script -q -c "$SCRIPT 2>&1" /dev/null | tee $LOG
    script -q -c "$SCRIPT 2>&1" /dev/null
fi
timer_end
