import pandas as pd
import redis
import argparse
import numpy as np
import usearch
from usearch.index import Index, MetricKind, MetricSignature, CompiledMetric

def read_data_from_hdf5():
    """Read data from HDF5 file and create three dataframes
    
    Args:
        test_num (int, optional): Number of rows to read. If None, reads all rows.
        
    Returns:
        tuple: (train_data, test_data, neighbors) as pandas DataFrames
    """
    import h5py
    
    file_path = 'data/deep-image-96-angular.hdf5'
    
    # Use h5py to read the file
    with h5py.File(file_path, 'r') as f:
        # Print the keys to understand the structure
        print(f"HDF5 file keys: {list(f.keys())}")
        
        # Load datasets
        # Load datasets from HDF5 file and convert to numpy arrays
        train_dataset = np.array(f['train'])
        test_dataset = np.array(f['test'])
        neighbors_dataset = np.array(f['neighbors'])

    
    # Print summary
    print(f"Read {len(train_dataset)} rows of training data")
    print(f"Read {len(test_dataset)} rows of test data")
    print(f"Read {len(neighbors_dataset)} rows of neighbor data")
    # Show samples
    # print("\nSample train data:")
    # print(train_data.head())
    # print("\nSample test data:")
    # print(test_data.head())
    # print("\nSample neighbors data:")
    # print(neighbors.head())
    
    return train_dataset, test_dataset, neighbors_dataset

def test_recall_usearch(enable_batch_add):
    added_data, test_data, neighbors = read_data_from_hdf5()
    # create usearch index
    dimensions = len(added_data[0])
    print(f"Creating USearch index with {dimensions} dimensions")
    
    # Create USearch index with appropriate dimensions and metric
    index = Index(
        ndim=dimensions,
        metric=MetricKind.L2sq,
        dtype='f32'  # 32-bit float
    )
    
    if enable_batch_add:
        # Add vectors to Usearch concurrently
        index.add(
            keys=list(range(len(added_data))),
            vectors=added_data,
        )
        print(f"Added {len(added_data)} vectors to USearch index using batch add")
    else:
        # Add vectors to USearch index
        for i, vector in enumerate(added_data):
            index.add(i, vector)
            if i % 10000 == 0:
                print(f"Added {i} vectors to USearch index")
        
    # Search for nearest neighbors
    recall_sum = 0
    for i, query in enumerate(test_data):
        results = index.search(query, 100)
        # Extract the IDs from results
        result_ids = [match.key for match in results]
        
        # Calculate recall  
        true_neighbors = set(neighbors[i])
        found_neighbors = set(result_ids).intersection(true_neighbors)
        recall = len(found_neighbors) / len(true_neighbors)
        recall_sum += recall
        
        if i % 1000 == 0:
            print(f"Processed {i} queries, current avg recall: {recall_sum/(i+1)}")
    
    # Print final results
    avg_recall = recall_sum / len(test_data)
    print(f"Average recall: {avg_recall}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-batch-add", type=bool, default=True, help="Whether to enable batch add")
    args = parser.parse_args()
    
    test_recall_usearch(args.enable_batch_add);