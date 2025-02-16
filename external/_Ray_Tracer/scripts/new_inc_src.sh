#!/bin/bash

if [ "$#" -eq 0 ]; then
    exit 0
    
elif [ "$#" -eq 1 ]; then
    touch inc/$1.h
    touch src/$1.cpp
    
elif [ "$#" -eq 2 ]; then
    cp inc/$2.h    inc/$1.h
    cp src/$2.cpp  src/$1.cpp
fi

