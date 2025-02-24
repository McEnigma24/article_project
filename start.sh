#!/bin/bash

var_start=""
var_end=""
dir_input="input"
dir_build="build"
dir_exe="exe"
dir_log="log"

SCRIPT="./run.sh"
LOG="../output/start.log"

CALLING_ARGUMENT="$1"

dir_run_time_config="run_time_config"
path_DONE_installed="${dir_run_time_config}/DONE_installed.txt"

silent_come_back() { cd - > /dev/null; }

clear_file() { if [ -f $1 ]; then rm -rf $1; fi; }
clear_file_nested() { cd $1; clear_file "$2"; silent_come_back; }

clear_dir() { if [ -d $1 ]; then rm -rf $1; fi; mkdir $1; }
clear_dir_nested() { cd $1; clear_dir "$2"; silent_come_back; }

create_dir() { if [ ! -d $1 ]; then mkdir $1; fi; }
create_dir_nested() { cd $1; create_dir "$2"; silent_come_back; }

timer_start() { var_start=$(date +%s); }
timer_end() { var_end=$(date +%s); }

timer_print()
{
    elapsed=$((var_end - var_start))
    hours=$((elapsed / 3600))
    minutes=$(( (elapsed % 3600) / 60 ))
    seconds=$((elapsed % 60))
    printf "Program          - took: %02d:%02d:%02d\n" $hours $minutes $seconds
}
install_packages()
{
    if [ -f $path_DONE_installed ]; then return; fi

    # Funkcja sprawdzająca czy pakiet jest zainstalowany
    check_and_install()
    {
        PACKAGE=$1
        if ! dpkg-query -W -f='${Status}' "$PACKAGE" 2>/dev/null | grep "install ok installed" > /dev/null; then
            echo "$PACKAGE is not installed. Installing..."

            sudo apt install -y "$PACKAGE"

            if [ $? -eq 0 ]; then
                echo ""
            else
                echo -e "\nstart_all.sh - ERROR - unable to install this package: $PACKAGE\n"
                exit
            fi
        fi
    }

    # Lista pakietów do zainstalowania
    PACKAGES=(
        tar
        make
        cmake
        build-essential
        gcc
        g++
        libstdc++-11-dev
        gcc-multilib
        g++-multilib
        libssl-dev
        libcairo2-dev
        libx264-dev
        libswscale-dev
        libavcodec-dev
        libavutil-dev
        libavformat-dev
        librsvg2-dev
        libgdk-pixbuf2.0-dev
        libgtk-3-dev
        libavfilter-dev
    )

    # Aktualizacja listy pakietów
    sudo apt update -y > /dev/null 2>&1 && sudo apt upgrade -y > /dev/null 2>&1

    # Sprawdzanie i instalowanie każdego pakietu
    for PACKAGE in "${PACKAGES[@]}"; do
        check_and_install "$PACKAGE"
    done


    echo -ne "\n\n"
    echo "Instalation completed"
    echo "DONE" > $path_DONE_installed
    echo -ne "\n\n"
}
env_prep()
{
    create_dir "$dir_input"
    create_dir "$dir_build"
    create_dir "$dir_exe"
    create_dir "$dir_log"
    create_dir "$dir_run_time_config"

    if [[ "$CALLING_ARGUMENT" == *"-c"* ]]; then
        clear_dir "$dir_build"
    fi

    chmod +x scripts/*.sh

    clear
    cd scripts
}

#####################   START   #####################

install_packages

env_prep

timer_start
{
    script -q -c "$SCRIPT 2>&1" /dev/null | tee $LOG
}
timer_end

timer_print
