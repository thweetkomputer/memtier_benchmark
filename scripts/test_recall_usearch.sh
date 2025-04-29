#!/bin/bash
source ./scripts/config.sh

# python ./scripts/generate_data.py --num_vectors $1 --warm_size 0

python ./scripts/test_recall.py --add_num $ADD_NUM --test_num $TEST_NUM --k 100 --test_usearch=True
