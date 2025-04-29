#!/usr/bin/env python3
"""
EloqVec API Test Script

This script tests the Redis vector collection API functionality including:
- Creating vector collections
- Adding vectors
- Searching vectors
- Deleting vectors
- Dropping collections
"""

import redis
import sys


def log_error(message, expected, actual=None):
    """Print formatted error message"""
    if actual:
        print(f"ERROR: {message}. Expected: '{expected}', Actual: '{actual}'")
    else:
        print(f"ERROR: {message}")
    print("Test Failed")
    return False


def test_create_vec(redis_client):
    """Test vector collection creation functionality"""
    # Initial cleanup - drop collection if exists
    try:
        redis_client.execute_command('dropcollection', 'vector')
    except Exception as err:
        if str(err) != 'Collection vector not found':
            return log_error("Unexpected error during initial cleanup", 
                           'Collection vector not found', str(err))
    
    # Create new collection
    result = redis_client.execute_command('createcollection', 'vector', 2)
    if result != 'OK':
        return log_error("Failed to create collection", 'OK', result)
    
    # Test syntax error case
    try:
        redis_client.execute_command('createcollection', 'vector')
    except Exception as err:
        if str(err) != 'syntax error':
            return log_error("Unexpected error with missing dimension", 
                           'syntax error', str(err))
    
    # Test existing collection case
    try:
        redis_client.execute_command('createcollection', 'vector', 100)
    except Exception as err:
        if str(err) != 'Collection already exists':
            return log_error("Unexpected error with existing collection", 
                           'Collection already exists', str(err))
    
    # Test invalid dimension format
    try:
        redis_client.execute_command('createcollection', 'vector_2', 'a')
    except Exception as err:
        if str(err) != 'Invalid dimension format. Dimension must be a valid integer':
            return log_error("Unexpected error with invalid dimension", 
                           'Invalid dimension format. Dimension must be a valid integer', str(err))

    # All tests passed
    return True


def test_add_vec(redis_client):
    """Test vector addition functionality"""
    # Add first vector
    result = redis_client.execute_command('addvec', 'vector', '1', '1:1')
    if result != 'OK':
        return log_error("Failed to add first vector", 'OK', result)
    
    # Add second vector
    result = redis_client.execute_command('addvec', 'vector', '2', '2:2')
    if result != 'OK':
        return log_error("Failed to add second vector", 'OK', result)
    
    # Test syntax error case
    try:
        redis_client.execute_command('addvec', 'vector', '1')
    except Exception as err:
        if str(err) != 'syntax error':
            return log_error("Unexpected error with missing vector", 
                           'syntax error', str(err))
    
    # Test duplicate key case
    try:
        redis_client.execute_command('addvec', 'vector', '1', '1:2')
    except Exception as err:
        if str(err) != 'Key 1 already exists':
            return log_error("Unexpected error with duplicate key", 
                           'Key 1 already exists', str(err))
    
    # Test invalid key format
    try:
        redis_client.execute_command('addvec', 'vector', 'a', '1:2')
    except Exception as err:
        if str(err) != 'Invalid key format. Key must be a valid integer':
            return log_error("Unexpected error with invalid key format", 
                           'Invalid key format. Key must be a valid integer', str(err))
    
    # Test non-existent collection
    try:
        redis_client.execute_command('addvec', 'vector_2', '1', '1:1')
    except Exception as err:
        if str(err) != 'Collection vector_2 not found':
            return log_error("Unexpected error with non-existent collection", 
                           'Collection vector_2 not found', str(err))


    # All tests passed
    return True


def test_search_vec(redis_client):
    """Test vector search functionality"""
    # Search for first vector
    result = redis_client.execute_command('searchvec', 'vector', '1', '1:1')
    if len(result) != 1:
        return log_error("Unexpected result length for first search", 1, len(result))
    
    if result[0] != '| 1 | 0.000000 |':
        return log_error("Unexpected result content for first search", 
                        '| 1 | 0.000000 |', result[0])
    
    # Search with K=2 to find more results
    result = redis_client.execute_command('searchvec', 'vector', '2', '1:1')
    if len(result) != 2:
        return log_error("Unexpected result length for second search", 2, len(result))
    
    if result[0] != '| 1 | 0.000000 |':
        return log_error("Unexpected first result for second search", 
                        '| 1 | 0.000000 |', result[0])
    
    if result[1] != '| 2 | 2.000000 |':
        return log_error("Unexpected second result for second search", 
                        '| 2 | 2.000000 |', result[1])
    
    # Test syntax error case
    try:
        redis_client.execute_command('searchvec', 'vector', '1')
    except Exception as err:
        if str(err) != 'syntax error':
            return log_error("Unexpected error with missing vector parameter", 
                           'syntax error', str(err))
    
    # Test invalid vector format
    try:
        redis_client.execute_command('searchvec', 'vector', '1', 'sa')
    except Exception as err:
        if str(err) != 'Parse failed: Invalid value of vector':
            return log_error("Unexpected error with invalid vector format", 
                           'Parse failed: Invalid value of vector', str(err))
    
    # Test invalid K value
    try:
        redis_client.execute_command('searchvec', 'vector', 'a', '1:1')
    except Exception as err:
        if str(err) != 'Invalid K value format. K must be a valid integer':
            return log_error("Unexpected error with invalid K value", 
                           'Invalid K value format. K must be a valid integer', str(err))

    # All tests passed
    return True


def test_delete_vec(redis_client):
    """Test vector deletion functionality"""
    # Delete first vector
    result = redis_client.execute_command('delvec', 'vector', '1')
    if result != 'OK':
        return log_error("Failed to delete vector 1", 'OK', result)
    
    # Verify only vector 2 remains
    result = redis_client.execute_command('searchvec', 'vector', '2', '1:1')
    if len(result) != 1:
        return log_error("Unexpected result length after deletion", 1, len(result))
    
    if result[0] != '| 2 | 2.000000 |':
        return log_error("Unexpected result content after deletion", 
                        '| 2 | 2.000000 |', result[0])
    
    # Test adding with deleted key
    try:
        redis_client.execute_command('addvec', 'vector', '1', '1:1')
    except Exception as err:
        expected = 'Key 1 has been deleted. Please use another key to identify the vector'
        if str(err) != expected:
            return log_error("Unexpected error when using deleted key", expected, str(err))
    
    # Delete second vector
    result = redis_client.execute_command('delvec', 'vector', '2')
    if result != 'OK':
        return log_error("Failed to delete vector 2", 'OK', result)
    
    # Verify collection is empty
    result = redis_client.execute_command('searchvec', 'vector', '2', '1:1')
    if result != 'Collection is empty':
        return log_error("Collection should be empty after deleting all vectors", 
                        'Collection is empty', result)
    
    # Test non-existent collection
    try:
        redis_client.execute_command('delvec', 'vector_2', '2')
    except Exception as err:
        if str(err) != 'Collection vector_2 not found':
            return log_error("Unexpected error with non-existent collection", 
                           'Collection vector_2 not found', str(err))

    # All tests passed
    return True


def test_drop_collection(redis_client):
    """Test collection drop functionality"""
    # Drop the collection
    result = redis_client.execute_command('dropcollection', 'vector')
    if result != 'OK':
        return log_error("Failed to drop collection", 'OK', result)
    
    # Test dropping non-existent collection
    try:
        redis_client.execute_command('dropcollection', 'vector')
    except Exception as err:
        if str(err) != 'Collection vector not found':
            return log_error("Unexpected error when dropping non-existent collection", 
                           'Collection vector not found', str(err))
    
    # Verify vector search fails after dropping
    try:
        redis_client.execute_command('searchvec', 'vector', 1, '1:1')
    except Exception as err:
        if str(err) != 'Collection vector not found':
            return log_error("Unexpected error when searching dropped collection", 
                           'Collection vector not found', str(err))
    
    # Recreate the collection
    result = redis_client.execute_command('createcollection', 'vector', 2)
    if result != 'OK':
        return log_error("Failed to recreate collection", 'OK', result)
    
    # Verify empty collection
    result = redis_client.execute_command('searchvec', 'vector', 1, '1:1')
    if result != 'Collection is empty':
        return log_error("Collection should be empty after recreation", 
                        'Collection is empty', result)
    
    # Add vector to new collection
    result = redis_client.execute_command('addvec', 'vector', 1, '1:1')
    if result != 'OK':
        return log_error("Failed to add vector to new collection", 'OK', result)
    
    # Verify vector was added
    result = redis_client.execute_command('searchvec', 'vector', 1, '1:1')
    if result[0] != '| 1 | 0.000000 |':
        return log_error("Unexpected search result after adding vector", 
                        '| 1 | 0.000000 |', result[0])
    
    # Drop collection again
    result = redis_client.execute_command('dropcollection', 'vector')
    if result != 'OK':
        return log_error("Failed to drop collection second time", 'OK', result)
    
    # Recreate collection again
    result = redis_client.execute_command('createcollection', 'vector', 2)
    if result != 'OK':
        return log_error("Failed to recreate collection second time", 'OK', result)
    
    # Verify empty collection
    result = redis_client.execute_command('searchvec', 'vector', 1, '1:1')
    if result != 'Collection is empty':
        return log_error("Collection should be empty after second recreation", 
                        'Collection is empty', result)
    
    # Add vector to new collection
    result = redis_client.execute_command('addvec', 'vector', 1, '1:1')
    if result != 'OK':
        return log_error("Failed to add vector to new collection second time", 'OK', result)


    # All tests passed
    return True


def test_eloqvec(redis_host, redis_port):
    """Main test function - runs all EloqVec API tests"""
    print(f"\n{'='*50}\nRunning EloqVec API Tests\n{'='*50}")
    
    # Connect to Redis
    try:
        redis_client = redis.Redis(redis_host, redis_port, decode_responses=True)
        redis_client.ping()
        print(f"✓ Successfully connected to Redis at {redis_host}:{redis_port}\n")
    except redis.ConnectionError:
        print(f"✗ Failed to connect to Redis at {redis_host}:{redis_port}")
        print("Test Failed")
        return False
    
    # Run test suite
    tests = [
        ("Vector Collection Creation", test_create_vec),
        ("Vector Addition", test_add_vec),
        ("Vector Search", test_search_vec),
        ("Vector Deletion", test_delete_vec),
        ("Collection Management", test_drop_collection),
    ]
    
    for test_name, test_func in tests:
        print(f"Testing: {test_name}...")
        if not test_func(redis_client):
            print(f"✗ {test_name} test failed")
            return False
        print(f"✓ {test_name} test passed\n")
    
    return True


def main():
    """Script entry point"""
    host = '127.0.0.1'
    port = 6380
    
    # Allow command-line arguments for host and port
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    # Run tests
    res = test_eloqvec(host, port)
    
    # Print final result
    print(f"\n{'='*50}")
    if res:
        print("✓ All tests passed successfully!")
    else:
        print("✗ Test suite failed")
    print(f"{'='*50}\n")
    
    # Exit with appropriate status code
    sys.exit(0 if res else 1)


if __name__ == "__main__":
    main()