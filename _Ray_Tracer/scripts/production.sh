#!/bin/bash

DONT_CLEAR="$1"
DIR_ROOT=$(dirname "$(pwd)")
DIR_BUILD="build"
DIR_LOG="log"
DIR_TARGET="exe"

remove_single_file() { [ -f "$1" ] && rm "$1"; }
clear_dir() { if [ -d $1 ]; then rm -rf $1; fi; mkdir $1; }
clear_dir_with_extension() { if ls $1/*$2 1> /dev/null 2>&1; then rm -f $1/*$2; fi }

clean_env() { cd $DIR_ROOT; echo -e "\nBuild (1/4) - Cleaning env"; clear_dir "$DIR_TARGET"; clear_dir_with_extension "$DIR_BUILD" ".exe"; }
prep_env()
{
    cd $DIR_ROOT
    echo -e "\nBuild (2/4) - Preparing env"

    cd src
    find ./ -name "*.cpp" -exec sha256sum {} + > tmp_checksum.txt

    if [ ! -f __checksums__.txt ]; then
        # Nie ma poprzedniego pliku checksum, kopiujemy wszystko
        echo "              Copying to CUDA"
        clear_dir "CUDA"
        cp *.cpp CUDA
        cd CUDA
        for file in *.cpp; do mv -- "$file" "${file%.cpp}.cu"; done
        
        # Zapisanie checksum dla każdego pliku
        cd ..
        cp tmp_checksum.txt __checksums__.txt
    else
        # Porównujemy nowe checksumy z poprzednimi
        changed_files=$(comm -13 <(sort __checksums__.txt) <(sort tmp_checksum.txt) | awk '{print $2}')

        if [ ! -z "$changed_files" ]; then
            echo "              Copying changed files to CUDA"
            
            for file in $changed_files; do
                echo "              -$(basename "$file")"
                cp "$file" CUDA/
                mv "CUDA/$(basename "$file")" "CUDA/$(basename "$file" .cpp).cu"
            done

            # Aktualizacja checksum
            cp tmp_checksum.txt __checksums__.txt
        else
            echo "              No changes detected, skipping copy"
        fi
    fi

    rm tmp_checksum.txt
}
build_all() { cd $DIR_ROOT; echo -e "\nBuild (3/4) - Building"; cd $DIR_BUILD; cmake .. > /dev/null 2>&1; make -j$(nproc); cd - > /dev/null 2>&1; }
copy_exe()  { cd $DIR_ROOT; echo -e "\nBuild (4/4) - Copying to exe"; cp $DIR_BUILD/*.exe $DIR_TARGET; }


# START #

if [ "$DONT_CLEAR" != "log_to_terminal" ]; then clear; fi

clean_env

# prep_env

build_all

copy_exe
