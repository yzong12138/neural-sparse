import argparse
import time
import numpy as np
import random
import sys
import os
import multiprocessing
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from scripts.local_embedding_client import LocalEmbeddingService, SimpleLocalEmbeddingClient


def run_server_process(directory, model_id, port):
    """Run the embedding server in a separate process"""
    os.system(f"python scripts/local_embedding_server.py --directory {directory} --model_id {model_id} --port {port}")


def run_load_test(url, num_requests, batch_size, concurrency):
    """Run a load test against the embedding server"""
    print(f"Running load test with {num_requests} requests, batch size {batch_size}, {concurrency} concurrent threads")
    
    # Generate random text IDs for testing
    all_ids = [random.randint(1, 10000) for _ in range(num_requests * batch_size)]
    
    # Split into batches
    id_batches = [all_ids[i:i+batch_size] for i in range(0, len(all_ids), batch_size)]
    
    # Create a simple client
    client = SimpleLocalEmbeddingClient(server_url=url)
    
    # Function to process one batch
    def process_batch(batch_ids):
        try:
            start_time = time.time()
            result = client.get_embeddings(batch_ids)
            elapsed = time.time() - start_time
            return elapsed, len(result), [1 if emb is not None else 0 for emb in result].count(1)
        except Exception as e:
            print(f"Error processing batch: {e}")
            return None
    
    # Process batches with thread pool
    results = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(id_batches)}
        for future in tqdm(future_to_batch, desc="Processing batches"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    if results:
        total_embeddings = sum(found for _, _, found in results)
        total_requests = len(results)
        avg_time = sum(time for time, _, _ in results) / len(results)
        
        print("\nLoad Test Results:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Total requests: {total_requests}")
        print(f"Total embeddings found: {total_embeddings}")
        print(f"Average request time: {avg_time:.4f}s")
        print(f"Requests per second: {total_requests/total_time:.2f}")
        print(f"Embeddings per second: {total_embeddings/total_time:.2f}")
    else:
        print("No valid results received")


def async_client_example(server_url, text_ids):
    """Example of using the async client"""
    print("\nAsync Client Example:")
    client = LocalEmbeddingService(server_url=server_url)
    
    # Register the task
    print("Registering task...")
    client.register_task(text_ids)
    
    # Fetch the embedding
    print("Fetching embedding...")
    start_time = time.time()
    embeddings = client.fetch_embedding(text_ids)
    elapsed = time.time() - start_time
    
    print(f"Fetched {len(embeddings)} embeddings in {elapsed:.4f}s")
    print(f"Embeddings shape: {embeddings[0].shape if len(embeddings) > 0 and embeddings[0] is not None else 'N/A'}")
    
    # Check server health
    health = client.health_check()
    print(f"Server health: {health}")
    
    # Clean up
    client.shutdown()


def simple_client_example(server_url, text_ids):
    """Example of using the simple client"""
    print("\nSimple Client Example:")
    client = SimpleLocalEmbeddingClient(server_url=server_url)
    
    # Fetch embeddings
    print("Fetching embeddings...")
    start_time = time.time()
    embeddings = client.get_embeddings(text_ids)
    elapsed = time.time() - start_time
    
    print(f"Fetched {len(embeddings)} embeddings in {elapsed:.4f}s")
    print(f"Embeddings shape: {embeddings[0].shape if len(embeddings) > 0 and embeddings[0] is not None else 'N/A'}")


def main():
    parser = argparse.ArgumentParser(description="Embedding Server Example")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing embedding files")
    parser.add_argument("--model_id", type=int, default=0, help="Model ID")
    parser.add_argument("--port", type=int, default=8000, help="Port for the server")
    parser.add_argument("--start_server", action="store_true", help="Start the server as part of this script")
    parser.add_argument("--run_load_test", action="store_true", help="Run a load test against the server")
    parser.add_argument("--num_requests", type=int, default=1000, help="Number of requests for load test")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for load test")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency level for load test")
    args = parser.parse_args()
    
    server_url = f"http://localhost:{args.port}"
    
    # Start server if requested
    server_process = None
    if args.start_server:
        print(f"Starting embedding server on port {args.port}...")
        server_process = multiprocessing.Process(
            target=run_server_process, 
            args=(args.directory, args.model_id, args.port)
        )
        server_process.start()
        
        # Give the server time to start
        print("Waiting for server to start...")
        time.sleep(5)
    
    try:
        # Check if server is running
        try:
            response = requests.get(f"{server_url}/health")
            if response.status_code == 200:
                print("Server is running")
                server_info = response.json()
                print(f"Server info: {server_info}")
            else:
                print(f"Server returned status {response.status_code}")
                return
        except requests.exceptions.ConnectionError:
            print(f"Could not connect to server at {server_url}")
            return
        
        # Run examples
        # Use some sample text IDs for testing
        sample_text_ids = [1, 2, 3, 4, 5]
        
        # Run the Simple client example
        simple_client_example(server_url, sample_text_ids)
        
        # Run the Async client example
        async_client_example(server_url, sample_text_ids)
        
        # Run load test if requested
        if args.run_load_test:
            run_load_test(server_url, args.num_requests, args.batch_size, args.concurrency)
    
    finally:
        # Shut down the server if we started it
        if server_process is not None:
            print("Shutting down server...")
            server_process.terminate()
            server_process.join()
            print("Server shut down")


if __name__ == "__main__":
    main()
