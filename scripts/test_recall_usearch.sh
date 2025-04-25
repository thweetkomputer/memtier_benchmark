#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Please provide a number of vectors as an argument."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: Please provide a number of needed neighbors as an argument."
    exit 1
fi

K=$2

# python ./scripts/generate_data.py --num_vectors $1 --warm_size 0

python ./scripts/test_recall.py --add_num $1 --k $K --test_usearch=True
