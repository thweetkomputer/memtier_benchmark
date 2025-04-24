#!/bin/bash

# Define parameters
THREADS=$1

if [ -z "$THREADS" ]; then
    echo "Error: Please provide a number of threads as an argument."
    exit 1
fi

CLIENTS=1
DATA_FILE="./data/search_vectors_data.csv"

# Calculate total number of clients for main stage
TOTAL_CLIENTS=$((THREADS * CLIENTS))

# Count the number of lines in the data files
TOTAL_LINES=$(wc -l "$DATA_FILE" | awk '{print $1}')
# Account for header line
TOTAL_LINES=$((TOTAL_LINES - 1))

# Calculate requests per client for each stage
REQUESTS=$((TOTAL_LINES / TOTAL_CLIENTS))
REDIS_SERVER=192.168.122.33
K=10

echo "=== Main Stage Settings ==="
echo "Main data file: $DATA_FILE"
echo "Main data lines: $TOTAL_LINES"
echo "Total clients: $TOTAL_CLIENTS (threads=$THREADS × clients=$CLIENTS)"
echo "Requests per client: $REQUESTS"

# Run main stage with calculated parameters
echo ""
echo "=== Starting Main Stage ==="
./memtier_benchmark \
    --server=$REDIS_SERVER \
    --port=6380 \
    --protocol=redis \
    --command="searchvec vector_table $K __data__" \
    --data-import="$DATA_FILE" \
    --threads=$THREADS \
    --clients=$CLIENTS \
    --requests=$REQUESTS
