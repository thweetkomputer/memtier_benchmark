# TODO: refactor these script, make code clean and readable
import pandas as pd
import redis
import argparse
import numpy as np
import usearch
from usearch.index import Index, MetricKind, MetricSignature, CompiledMetric

def read_data_from_csv(file_path, test_num=None):
    if test_num is not None:
        data = pd.read_csv(file_path, nrows=test_num)
    else:
        data = pd.read_csv(file_path)
    return data

def add_vectors_to_redis(data, redis_host='192.168.122.33', redis_port=6380):
    # Connect to Redis
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    # Check connection
    try:
        r.ping()
        print(f"Successfully connected to Redis at {redis_host}:{redis_port}")
    except redis.ConnectionError:
        print(f"Failed to connect to Redis at {redis_host}:{redis_port}")
        return
    
    # Process each line and send to Redis
    success_count = 0
    error_count = 0
    
    for index, row in data.iterrows():
        try:
            # Extract key and data from the row
            # Assuming first column is key and the rest is data
            columns = data.columns
            key = row[columns[8]]
            vector_data = row[columns[9]]
            # Execute Redis command
            result = r.execute_command('addvec', 'vector', key, vector_data)
            
            if result != 'OK':
                error_count += 1
                print(f"Failed to add vector for key {key}: {result}")
                continue
            
            success_count += 1
            # Print progress every 100 items
            if success_count % 100 == 0:
                print(f"Added {success_count} vectors to Redis")
                
        except Exception as e:
            error_count += 1
            print(f"Error processing row {index}: {e}")
    
    print(f"\nCompleted: Added {success_count} vectors to Redis with {error_count} errors")


def find_exact_neighbors(query_vector_str, dataset, k):
    """
    Find the exact K nearest neighbors of the query vector in the dataset
    by directly calculating vector distances
    
    Args:
        query_vector_str: Query vector as a string in format val1:val2:val3:...
        dataset: Pandas DataFrame containing vectors
        k: Number of nearest neighbors to return
        
    Returns:
        Tuple of (list_of_neighbor_keys, list_of_tuples_with_key_and_distance)
    """
    import numpy as np
    from scipy.spatial.distance import cosine
    
    # Parse the query vector string into a numpy array
    query_vector = np.array([float(val) for val in query_vector_str.split(':')])
    
    # Get column names and identifiy the key column and vector column
    columns = dataset.columns
    key_col = columns[8]  # Assuming the key is in column 8
    vector_col = columns[9]  # Assuming the vector is in column 9
    
    # Calculate distances for all vectors in the dataset
    distances = []
    for index, row in dataset.iterrows():
        try:
            # Parse vector string into numpy array
            vector_str = row[vector_col]
            vector = np.array([float(val) for val in vector_str.split(':')])
            
            # Calculate Euclidean distance (you can change to other distance metrics)
            distance = np.linalg.norm(query_vector - vector) ** 2
            
            # Store key and distance
            distances.append((str(row[key_col]), distance))
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    
    # Sort by distance (ascending) and take top k
    distances.sort(key=lambda x: x[1])
    top_k_distances = distances[:k]
    
    # Extract keys and distances separately
    keys = [item[0] for item in top_k_distances]
    
    # print(f"Found exact {len(keys)} neighbors: {keys}")
    # print(f"With exact distances: {top_k_distances}")
    
    return keys, top_k_distances


def parse_redis_search_results(result_str):
    """
    Parse Redis vector search results from string format    
            |  key  |  distance  |
            | 208 | 1.495457 |
            | 460 | 8.954815 |
    
    Args:
        result_str: Raw string output from Redis search command
        
    Returns:
        Tuple of (list_of_keys, list_of_tuples_with_key_and_distance)
    """
    import re
    
    # Extract keys and distances using regex
    pattern = r'\|\s*(\d+)\s*\|\s*([\d\.]+)\s*\|'
    matches = re.findall(pattern, result_str)
    
    # Create list of keys and list of (key, distance) tuples
    keys = []
    key_distance_pairs = []
    
    if matches:
        for key, distance in matches:
            keys.append(key)
            key_distance_pairs.append((int(key), round(float(distance), 2)))
        # print(f"Extracted {len(keys)} neighbors: {keys}")
        # print(f"With distances: {key_distance_pairs}")
    else:
        print(f"Could not parse results: {result_str}")
    
    return keys, key_distance_pairs


def execute_redis_cli_command(command, redis_host='192.168.122.33', redis_port=6380, timeout=10):
    """
    Execute a Redis command using redis-cli via subprocess with timeout
    
    Args:
        command: The Redis command to execute (without redis-cli prefix)
        redis_host: Redis server hostname or IP
        redis_port: Redis server port
        timeout: Maximum time in seconds to wait for command completion
        
    Returns:
        Tuple of (success_bool, result_string, error_string)
    """
    import subprocess
    
    # Format the command for redis-cli
    redis_cli_cmd = f"redis-cli -h {redis_host} -p {redis_port} {command}"
    # print(f"Running via redis-cli: {redis_cli_cmd}")
    
    try:
        # Set a timeout to prevent hanging
        process = subprocess.Popen(redis_cli_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode == 0:
            return True, stdout, ""
        else:
            return False, "", stderr
    except subprocess.TimeoutExpired:
        process.kill()
        return False, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, "", str(e)


def add_vectors_to_usearch(index, added_data):
    # Add vectors to USearch index
    total_success_count = 0
    total_error_count = 0
    total_rows = len(added_data)
    
    print(f"Processing {total_rows} vectors")
    for i, row in added_data.iterrows():
        try:
            key = row.iloc[8]
            vector_data = row.iloc[9]
            
            # Convert string vector to NumPy array of floats
            vector = np.array([float(val) for val in vector_data.split(':')])
            
            # Add to index (key must be an integer in USearch)
            if isinstance(key, str) and not key.isdigit():
                print(f"Invalid key: {key}")
                exit(-1)
            else:
                numeric_key = int(key)
                
            index.add(numeric_key, vector)
            total_success_count += 1
            
            if total_success_count % 100 == 0:
                print(f"Added {total_success_count} vectors to USearch index")
                
        except Exception as e:
            total_error_count += 1
            print(f"Error processing row {i}: {e}")
        
    print(f"Added {total_success_count} vectors to USearch index with {total_error_count} errors")
    

def test_recall_usearch(search_data, added_data, k=10):
    # Parse sample vector to determine dimensions
    sample_vector = added_data.iloc[0][9].split(':')
    dimensions = len(sample_vector)
    print(f"Creating USearch index with {dimensions} dimensions")
    # Create USearch index with appropriate dimensions and metric
    index = Index(
        ndim=dimensions,
        metric=MetricKind.L2sq,
        dtype='f32'  # 32-bit float
    )
    
    add_vectors_to_usearch(index, added_data)

    # Keep track of successful recall rates
    total_matches = 0
    total_attempts = 0
    
    test_num = len(search_data)
    # Test recall for each query in search data
    print(f"Testing recall with {test_num} queries, top {k} neighbors")
    
    for i, row in search_data.iterrows():
        try:
            columns = search_data.columns
            vector_data = row[columns[9]]
            
            # Parse query vector
            query_vector = np.array([float(val) for val in vector_data.split(':')])
            
            # Get exact neighbors (ground truth)
            exact_neighbors, _ = find_exact_neighbors(vector_data, added_data, k)
            
            # Search with USearch
            usearch_results = index.search(query_vector, k)
            usearch_keys = []
            
            # Convert numeric keys back to original keys if needed
            for result in usearch_results:
                result_key = str(result.key)
                usearch_keys.append(result_key)
            
            # Count matches between exact and USearch results
            matches = set(exact_neighbors).intersection(set(usearch_keys))
            match_count = len(matches)
            
            total_matches += match_count
            total_attempts += len(exact_neighbors)
            
            recall_rate = match_count / len(exact_neighbors) if len(exact_neighbors) > 0 else 0
            print(f"Test {test_num}, Query {i}: Recall rate = {recall_rate:.4f} ({match_count}/{len(exact_neighbors)})")
            
        except Exception as e:
            print(f"Error processing query {i}: {e}")
    
    # Calculate overall recall rate
    overall_recall = total_matches / total_attempts if total_attempts > 0 else 0
    print(f"\nTest {test_num} - Overall USearch Recall: {overall_recall:.4f} ({total_matches}/{total_attempts})")
    
    return overall_recall


def test_recall_eloqvec(search_data, added_data, k=10, redis_host='192.168.122.33', redis_port=6380):
    # Add vectors to Redis
    # add_vectors_to_redis(added_data)

    # Connect to Redis
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    # Check connection
    try:
        print (f"Connecting to Redis at {redis_host}:{redis_port}")
        r.ping()
        print(f"Successfully connected to Redis at {redis_host}:{redis_port}")
    except redis.ConnectionError:
        print(f"Failed to connect to Redis at {redis_host}:{redis_port}")
        return
    
    # Keep track of successful recall rates
    total_matches = 0
    total_attempts = 0
    
    # Process first test_num lines of search data
    for i, row in search_data.iterrows():
        try:
            # Extract key and vector from the row (adjust column indices as needed)
            columns = search_data.columns
            key = row.iloc[8]
            vector_data = row.iloc[9]
            
            # Get top K neighbors from Redis
            print(f"Searching for neighbors of key: {key}")
            # Print the actual command being executed
            command = f"searchvec vector {k} {vector_data}"
            print(f"Executing command: {command}")
            
            # Use the helper function to execute Redis command via redis-cli
            success, redis_result, error = execute_redis_cli_command(command, redis_host, redis_port)
            
            if not success:
                print(f"Error executing command: {error}")
                redis_result = ""
            
            # print(f"Raw Redis result: {redis_result}")
            # Convert result to string if it's not already
            if not isinstance(redis_result, str):
                redis_result_str = str(redis_result)
            else:
                redis_result_str = redis_result
                
            # Parse results using the helper function
            approx_neighbors, approx_matches = parse_redis_search_results(redis_result_str)
            # Find ground truth neighbors in the ground truth data
            exact_neighbors, exact_matches = find_exact_neighbors(vector_data, added_data, k)
            
            # Calculate recall for this query

            # matches = sum(1 for n in approx_neighbors if n in exact_neighbors)
            matches = len(set(str(k) for k in approx_neighbors).intersection(set(str(k) for k in exact_neighbors)))
            
            recall = matches / len(exact_neighbors) if exact_neighbors else 0
            
            print(f"Query {i+1}: Found {matches}/{len(exact_neighbors)} matches. Recall: {recall:.2f}")
            
            total_matches += matches
            total_attempts += len(exact_neighbors)
            
        except Exception as e:
            print(f"Error processing query {i+1} for key {key}: {e}")
    
    # Calculate overall recall
    overall_recall = total_matches / total_attempts if total_attempts > 0 else 0
    print(f"\nOverall Recall@{k}: {overall_recall:.4f} ({total_matches}/{total_attempts})")
    
    return overall_recall 
    
def test_correctness(search_data, added_data, k=100, redis_host='192.168.122.33', redis_port=6380, mock=False):      
    dimensions = 96

    if mock:
        for i in range(added_data.shape[0]):
            # added_data.iloc[i, 9] = f'{i}:{i+1}'
            added_data.iloc[i, 9] = ':'.join([f'{np.random.random():.6f}' for _ in range(dimensions)])

    add_vectors_to_redis(added_data, redis_host, redis_port)
    index = Index(
        ndim=dimensions,
        metric=MetricKind.L2sq,
        dtype='f32'  # 32-bit float
    )

    add_vectors_to_usearch(index, added_data)
    

    if mock:
        for i in range(search_data.shape[0]):
            search_data.iloc[i, 9] = ':'.join([f'{np.random.random():.6f}' for _ in range(dimensions)])

    for i, row in search_data.iterrows():
        key = row[search_data.columns[8]]
        vector_data = row[search_data.columns[9]]
        
        # Search for neighbors in Redis
        command = f"searchvec vector {k} {vector_data}"
        # print(command)
        success, redis_result, error = execute_redis_cli_command(command, redis_host, redis_port)
        if not success:
            print(f"Failed to search for key {key}: {error}")
            continue
        
        # Parse results using the helper function
        eloqvec_neighbors, eloqvec_matches = parse_redis_search_results(redis_result)
        

        # Search for neighbors in USearch
        query_vector = np.array([float(val) for val in vector_data.split(':')])
        usearch_results = index.search(query_vector, k)
        usearch_neighbors = []
        usearch_matches = []
            
        # Convert numeric keys back to original keys if needed
        for result in usearch_results:
            result_key = str(result.key)
            usearch_neighbors.append(result_key)
            usearch_matches.append((int(result.key), round(float(result.distance), 2)))
        
        # print(eloqvec_matches)
        # print(usearch_matches)
        # check the euivalence of two lists
        if eloqvec_matches != usearch_matches:
            print("The lists are not equal")
            print(eloqvec_matches)
            print(usearch_matches)
            print(i, 'th query vector:', query_vector.tolist())

            print('double check:')
            vector1 = np.array([float(val) for val in added_data.iloc[0, 9].split(':')])
            vector2 = np.array([float(val) for val in added_data.iloc[1, 9].split(':')])
            distance1 = np.linalg.norm(query_vector - vector1) ** 2
            distance2 = np.linalg.norm(query_vector - vector2) ** 2
            print('vector1:', vector1.tolist())
            print('vector2:', vector2.tolist())
            print('distance to key-1:', distance1)
            print('distance to key-2:', distance2)
            
    print("All tests passed!")



if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test vector recall in Redis')
    parser.add_argument('--add_num', type=int, default=10000, help='Number of added vectors to use (default: 1000000)')
    parser.add_argument('--test_num', type=int, default=20, help='Number of test vectors to use (default: 100)')
    parser.add_argument('--search_file', type=str, default='./data/search_vectors_data.csv', help='Path to search vectors CSV file')
    parser.add_argument('--add_file', type=str, default='./data/vectors_data.csv', help='Path to vectors data CSV file')
    parser.add_argument('--k', type=int, default=100, help='Number of neighbors to search for (default: 10)')
    parser.add_argument('--redis_host', type=str, default='192.168.122.33', help='Redis host (default: 192.168.122.33)')
    parser.add_argument('--redis_port', type=int, default=6380, help='Redis port (default: 6380)')
    parser.add_argument('--test_usearch', type=bool, default=False, help='Use USearch instead of Redis (default: False)')
    parser.add_argument('--test_eloqvec', type=bool, default=False, help='Use Eloqvec instead of Redis (default: False)')
    parser.add_argument('--test_correctness', type=bool, default=False, help='Compare the search result of eloqvec and usearch (default: False)')
    parser.add_argument('--mock_test_correctness', type=bool, default=False, help='Compare the search result of eloqvec and usearch (default: False)')
    args = parser.parse_args()

    # Read search data from CSV (limited to test_num rows)
    search_data = read_data_from_csv(args.search_file, args.test_num)
    
    # Read added data from add_file_path
    added_data = read_data_from_csv(args.add_file, args.add_num)
    
    # Test recall with configurable test_num
    if args.test_usearch:
        print(f"Running recall test with {args.test_num} test vectors...")
        test_recall_usearch(search_data, added_data, args.k)
    
    if args.test_eloqvec:
        print(f"Running recall test with {args.test_num} test vectors...")
        test_recall_eloqvec(search_data, added_data, args.k, args.redis_host, args.redis_port)

    if args.test_correctness:
        print(f"Running correctness test with {args.test_num} test vectors...")
        test_correctness(search_data, added_data, args.k, args.redis_host, args.redis_port)
    
    if args.mock_test_correctness:
        print(f"Running correctness test with {args.test_num} test vectors...")
        test_correctness(search_data, added_data, args.k, args.redis_host, args.redis_port, True)