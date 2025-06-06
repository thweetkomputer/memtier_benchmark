#!/bin/bash

source ./scripts/config.sh

# Define parameters

if [ -z "$1" ]; then
    echo "Error: Please provide a number of threads as an argument."
    exit 1
fi

THREADS=$1
CLIENTS=1
DATA_FILE="./data/vectors_data.csv"
WARM_DATA_FILE="./data/warm_vectors_data.csv"

# Warmup parameters
WARMUP_THREADS=1
WARMUP_CLIENTS=1

# Calculate total number of clients for main stage
TOTAL_CLIENTS=$((THREADS * CLIENTS))

# Count the number of lines in the data files
WARM_TOTAL_LINES=$(wc -l "$WARM_DATA_FILE" | awk '{print $1}')
TOTAL_LINES=$(wc -l "$DATA_FILE" | awk '{print $1}')

# Account for header line
WARM_TOTAL_LINES=$((WARM_TOTAL_LINES - 1))
TOTAL_LINES=$((TOTAL_LINES - 1))

# Calculate requests per client for each stage
WARMUP_REQUESTS=$WARM_TOTAL_LINES
REQUESTS=$((TOTAL_LINES / TOTAL_CLIENTS))

# Create table
redis-cli -h $REDIS_SERVER -p $REDIS_PORT 'createcollection' 'vector' $DIM

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

# Run warmup stage with one client
echo ""
echo "=== Starting Warmup Stage ==="
./memtier_benchmark \
    --server=$REDIS_SERVER \
    --port=$REDIS_PORT \
    --protocol=redis \
    --command="addvec vector __key__ __data__" \
    --data-import="$WARM_DATA_FILE" \
    --threads=$WARMUP_THREADS \
    --clients=$WARMUP_CLIENTS \
    --requests=$WARMUP_REQUESTS

# Run main stage with calculated parameters
echo ""
echo "=== Starting Main Stage ==="
./memtier_benchmark \
    --server=$REDIS_SERVER \
    --port=$REDIS_PORT \
    --protocol=redis \
    --command="addvec vector __key__ __data__" \
    --data-import="$DATA_FILE" \
    --threads=$THREADS \
    --clients=$CLIENTS \
    --requests=$REQUESTS
