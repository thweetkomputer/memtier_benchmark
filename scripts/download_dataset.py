def download_dataset(url="http://ann-benchmarks.com/deep-image-96-angular.hdf5", output_path=None, chunk_size=8192):
    """
    Download a dataset from a URL with progress bar
    
    Args:
        url (str): URL of the dataset to download
        output_path (str): Path to save the downloaded file, defaults to filename from URL
        chunk_size (int): Chunk/batch size in bytes for downloading (default: 8192)
        
    Returns:
        str: Path to the downloaded file
    """
    import os
    import urllib.request
    import shutil
    
    try:
        from tqdm import tqdm
    except ImportError:
        print("tqdm not installed, progress bar won't be shown.")
        print("Install it using: pip install tqdm")
        tqdm = lambda x, **kwargs: x
    
    if output_path is None:
        output_path = os.path.join("data", os.path.basename(url))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Download the file with progress bar
    print(f"Downloading dataset from {url} to {output_path}...")
    
    # Create a request with headers that mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        # Open the url and get file size
        with urllib.request.urlopen(req) as response:
            file_size = int(response.info().get('Content-Length', -1))
            
            # Initialize progress bar
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(url)) as pbar:
                # Create a new request for the actual download
                download_req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(download_req) as source, open(output_path, 'wb') as output:
                    bytes_downloaded = 0
                    
                    # Read and write in chunks to show progress
                    while True:
                        chunk = source.read(chunk_size)
                        if not chunk:
                            break
                        
                        output.write(chunk)
                        bytes_downloaded += len(chunk)
                        pbar.update(len(chunk))
        
        print(f"Download complete: {output_path}")
        return output_path
    
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        if e.code == 403:
            print("Access forbidden. The server may require authentication or doesn't allow downloads.")
        elif e.code == 404:
            print("URL not found. The dataset may have been moved or deleted.")
        print(f"Please check if the dataset is available at: {url}")
        return None
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        print("Please check your internet connection and the URL.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


if __name__ == "__main__":
    import argparse
    
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Download a dataset with progress bar')
    parser.add_argument('--dataset', type=str, choices=['deep-image-96-angular', 'sift-1M'], 
                        help='The dataset to download')
    
    args = parser.parse_args()
    
    if args.dataset == 'deep-image-96-angular':
        args.url = "http://ann-benchmarks.com/deep-image-96-angular.hdf5"
    elif args.dataset == 'sift-1M':
        args.url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"

    # Download the dataset with the provided arguments
    download_dataset(url=args.url)
    