#!/bin/bash

# Define parameters
THREADS=8
CLIENTS=64
DATA_FILE="/home/yicw/code/memtier_benchmark/data/search_vectors_data.csv"

# Calculate total number of clients for main stage
TOTAL_CLIENTS=$((THREADS * CLIENTS))

# Count the number of lines in the data files
TOTAL_LINES=$(wc -l "$DATA_FILE" | awk '{print $1}')
# Account for header line
TOTAL_LINES=$((TOTAL_LINES - 1))

# Calculate requests per client for each stage
REQUESTS=$((TOTAL_LINES / TOTAL_CLIENTS))

K=10

echo "=== Warmup Stage Settings ==="
echo "Warm data file: $WARM_DATA_FILE"
echo "Warm data lines: $WARM_TOTAL_LINES"
echo "Warmup clients: $WARMUP_CLIENTS (threads=$WARMUP_THREADS × clients=$WARMUP_CLIENTS)"
echo "Warmup requests: $WARMUP_REQUESTS"
echo ""
echo "=== Main Stage Settings ==="
echo "Main data file: $DATA_FILE"
echo "Main data lines: $TOTAL_LINES"
echo "Total clients: $TOTAL_CLIENTS (threads=$THREADS × clients=$CLIENTS)"
echo "Requests per client: $REQUESTS"

# Run main stage with calculated parameters
echo ""
echo "=== Starting Main Stage ==="
memtier_benchmark \
    --server=127.0.0.1 \
    --port=6380 \
    --protocol=redis \
    --command="searchvec vector_table $K __data__" \
    --data-import="$DATA_FILE" \
    --threads=$THREADS \
    --clients=$CLIENTS \
    --requests=$REQUESTS