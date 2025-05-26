import numpy as np
import os
import argparse

def generate_vector(dim=96, normalized=True):
    """
    Generate a random vector of specified dimension with float32 data type.
    
    Args:
        dim (int): Dimension of the vector, default is 96
        normalized (bool): Whether to normalize the vector to unit length, default is False
        
    Returns:
        numpy.ndarray: A random vector of shape (dim,) with float32 data type
    """
    # Generate random values between 0 and 1
    vector = np.random.random(dim).astype(np.float32)
    
    # Normalize the vector to have unit length if requested
    if normalized:
        vector = vector / np.linalg.norm(vector)
    
    return vector


def save_vectors_to_file_for_memtier(vectors, output_file, offset=0):
    """
    Save vectors to a CSV file in memtier_benchmark compatible format
    
    Args:
        vectors (list): List of numpy arrays representing vectors
        output_file (str): Path to save the CSV file
        offset (int): Offset for key numbering, default is 0
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    with open(output_file, 'w') as f:
        # Write CSV header according to memtier_benchmark format - exact format is critical
        f.write("dumpflags, time, exptime, nbytes, nsuffix, it_flags, clsid, nkey, key, data\n")
        
        for i, vector in enumerate(vectors):
            # Serialize the vector to a single string (space-separated)
            serialized_vector = ":".join(map(str, vector))
            
            # Calculate the length of the serialized data (add 2 for \r\n at the end)
            data_len = len(serialized_vector)
            
            # Generate a key
            key = f"{i+1+offset}"
            key_len = len(key)
            
            # Write in memtier_benchmark format with proper spacing after commas
            # Note: Adding extra 4 bytes to nbytes and extra 2 bytes to nkey as per README.import
            f.write(f"0, 0, 0, {data_len + 2}, 0, 0, 0, {key_len}, {key}, {serialized_vector}\n")
    
    print(f"Saved {len(vectors)} vectors to CSV file {output_file} with header")


def generate_search_queries(vectors, num_queries, similarity_factor=0.8, noise_std=0.2):
    """
    Generate search query vectors that are similar to existing vectors for realistic search scenarios
    
    Args:
        vectors (list): List of existing vectors
        num_queries (int): Number of query vectors to generate
        similarity_factor (float): Factor to determine how similar the queries should be (0-1)
        noise_std (float): Standard deviation of noise to add
        
    Returns:
        list: List of query vectors
    """
    queries = []
    vector_count = len(vectors)
    
    for i in range(num_queries):
        # Select a random vector from the dataset
        base_idx = np.random.randint(0, vector_count)
        base_vector = vectors[base_idx]
        
        # Create a similar vector by adding noise
        noise = np.random.normal(0, noise_std, size=base_vector.shape).astype(np.float32)
        query_vector = similarity_factor * base_vector + (1 - similarity_factor) * noise
        
        # Ensure the query is within the same value range as original vectors (0-1)
        query_vector = np.clip(query_vector, 0, 1)
        
        queries.append(query_vector)
    
    return queries


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate and save vectors with float32 data type')
    parser.add_argument('--num_vectors', type=int, default=10000, help='Number of vectors to generate (default: 5000)')
    parser.add_argument('--dim', type=int, default=96, help='Dimension of each vector (default: 96)')
    parser.add_argument('--output', type=str, default='./data/vectors_data.csv', help='Output file path (default: ./data/vectors_data.csv)')
    parser.add_argument('--warm_output', type=str, default='./data/warm_vectors_data.csv', 
                        help='Output file path for warm data (first 5000 vectors) (default: ./data/warm_vectors_data.csv)')
    parser.add_argument('--warm_size', type=int, default=0, 
                        help='Number of vectors to save as warm data (default: 2000)')
    parser.add_argument('--search_output', type=str, default='./data/search_vectors_data.csv',
                        help='Output file path for search query vectors (default: ./data/search_vectors_data.csv)')
    parser.add_argument('--num_queries', type=int, default=0,
                        help='Number of search query vectors to generate (default: 0)')
    parser.add_argument('--similarity', type=float, default=0.8,
                        help='Similarity factor for search queries (0-1, default: 0.8)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize vectors to unit length')
    args = parser.parse_args()
    
    # Ensure we generate at least enough vectors for warm data
    num_vectors = max(args.num_vectors, args.warm_size)
    
    # Generate multiple vectors
    vectors = [generate_vector(dim=args.dim, normalized=args.normalize) for _ in range(num_vectors)]
    
    # Print information about the first vector
    if vectors:
        vec = vectors[0]
        print(f"Vector shape: {vec.shape}")
        print(f"Vector data type: {vec.dtype}")
        print(f"First 5 elements of first vector: {vec[:5]}")
    
    # Save the first 'warm_size' vectors to the warm data file
    warm_vectors = vectors[:args.warm_size]
    save_vectors_to_file_for_memtier(warm_vectors, args.warm_output)
    print(f"Saved first {args.warm_size} vectors as warm data")
    
    # Save only the non-warm vectors to the main output file
    non_warm_vectors = vectors[args.warm_size:]
    save_vectors_to_file_for_memtier(non_warm_vectors, args.output, args.warm_size)
    print(f"Saved {len(non_warm_vectors)} non-warm vectors to main data file")
    
    # Generate and save search query vectors
    if args.num_queries > 0:
        print(f"Generating {args.num_queries} search query vectors...")
        # Generate search queries based on the existing vectors
        query_vectors = generate_search_queries(
            vectors, 
            args.num_queries, 
            similarity_factor=args.similarity,
            noise_std=0.2
        )
        
        # Save search query vectors to file
        save_vectors_to_file_for_memtier(query_vectors, args.search_output)
        print(f"Saved {len(query_vectors)} search query vectors to {args.search_output}")
