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

ssh compute-3 "kill -9 \$(pgrep eloqvec) 2>/dev/null"

ssh compute-3 "cd ~/code/eloqvec && ./build/eloqvec -ip 192.168.122.33 --core_number=8" &

sleep 3

# Add vectors to Eloqvec
./scripts/add_vector.sh 30

python ./scripts/test_recall.py --k $K
