#!/bin/bash

TEST_FLAG="$1"
DIR_ROOT=$(dirname "$(pwd)")
DIR_BUILD="build"
DIR_LOG="log"
DIR_TARGET="exe"

remove_single_file() { [ -f "$1" ] && rm "$1"; }
clear_dir() { if [ -d $1 ]; then rm -rf $1; fi; mkdir $1; }
clear_dir_with_extension() { if ls $1/*$2 1> /dev/null 2>&1; then rm -f $1/*$2; fi }

clean_env() { cd $DIR_ROOT; echo -e "\nBuild (1/5) - Cleaning env"; clear_dir "$DIR_TARGET"; clear_dir_with_extension "$DIR_BUILD" ".exe"; }
prep_env()  { cd $DIR_ROOT; echo -e "\nBuild (2/5) - Preparing env"; }
build_all() { cd $DIR_ROOT; echo -e "\nBuild (3/5) - Building"; cmake -S . -B $DIR_BUILD; if ! cmake --build $DIR_BUILD; then exit; fi }
run_tests() { cd $DIR_ROOT; echo -e "\nBuild (4/5) - Testing"; cd $DIR_BUILD; if ! ctest; then exit; fi }
copy_exe()  { cd $DIR_ROOT; echo -e "\nBuild (5/5) - Copying to exe"; cp $DIR_BUILD/*.exe $DIR_TARGET; }


# START #

clean_env

prep_env

build_all

run_tests

copy_exe
