#!/bin/bash

cd ..
./start.sh

cd output/img
# find ./ -name "*.bmp" -exec sha256sum {} + > ../../cmp/reference_bmp_checksum.txt  # creating reference
find ./ -name "*.bmp" -exec sha256sum {} + > ../../cmp/tmp.txt
cd - > /dev/null

changed_files=$(comm -13 <(sort cmp/reference_bmp_checksum.txt) <(sort cmp/tmp.txt) | awk '{print $2}')

if [ ! -z "$changed_files" ]; then
    echo -e "\nERROR: there is a difference\n"
else
    echo -e "\nALL GOOD\n"
fi