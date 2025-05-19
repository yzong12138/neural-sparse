# Local Embedding Server

This is a high-performance local embedding server implementation that can load text embeddings into memory and serve them via HTTP requests. It's designed to serve 1000+ embedding requests per second with minimal latency.

## Components

1. `local_embedding_server.py` - The server that loads embeddings from binary files and serves them via HTTP
2. `local_embedding_client.py` - Client adapters for interacting with the server
3. `run_embedding_server_example.py` - Example script demonstrating usage and load testing

## Features

- Fast in-memory embedding lookups
- Asynchronous and synchronous client implementations
- Performance monitoring and metrics collection
- Batch processing of embedding requests
- Concurrent request handling with multiple workers
- HTTP API with FastAPI for high performance

## Installation

The server requires the following dependencies:

```
fastapi
uvicorn
numpy
requests
tqdm
```

You can install them with pip:

```bash
pip install fastapi uvicorn numpy requests tqdm
```

## Usage

### Starting the Server

Start the embedding server by running:

```bash
python scripts/local_embedding_server.py \
  --directory /path/to/embedding/directory \
  --model_id 0 \
  --port 8000 \
  --workers 4
```

Where:
- `directory` is the path to the directory containing the `queries` and `docs` subdirectories with embedding files
- `model_id` is the model identifier (corresponds to the directories)
- `port` is the HTTP port for the server (default: 8000)
- `workers` is the number of Uvicorn workers (default: 4)

### Client Usage

#### Synchronous Client

For simple, direct usage:

```python
from scripts.local_embedding_client import SimpleLocalEmbeddingClient

# Create a client
client = SimpleLocalEmbeddingClient(server_url="http://localhost:8000")

# Get embeddings for text IDs
text_ids = [1, 2, 3, 4, 5]
embeddings = client.get_embeddings(text_ids)

# Process the embeddings
for i, emb in enumerate(embeddings):
    if emb is not None:
        print(f"Text ID {text_ids[i]}: Found embedding with shape {emb.shape}")
    else:
        print(f"Text ID {text_ids[i]}: No embedding found")
```

#### Asynchronous Client

For more complex, asynchronous usage with background tasks:

```python
from scripts.local_embedding_client import LocalEmbeddingService

# Create a service instance
service = LocalEmbeddingService(server_url="http://localhost:8000", max_workers=10)

# Register a task for background processing
text_ids = [1, 2, 3, 4, 5]
service.register_task(text_ids)

# Later, fetch the embedding (will wait if not already processed)
embeddings = service.fetch_embedding(text_ids)

# Clean up when done
service.shutdown()
```

### Example Usage and Load Testing

The `run_embedding_server_example.py` script provides examples of both client implementations and includes a load testing capability.

Run a simple example:

```bash
python scripts/run_embedding_server_example.py \
  --directory /path/to/embedding/directory \
  --model_id 0 \
  --port 8000 \
  --start_server
```

Run with load testing:

```bash
python scripts/run_embedding_server_example.py \
  --directory /path/to/embedding/directory \
  --model_id 0 \
  --port 8000 \
  --start_server \
  --run_load_test \
  --num_requests 1000 \
  --batch_size 10 \
  --concurrency 20
```

## Server API Endpoints

The server provides the following HTTP endpoints:

- `POST /embeddings` - Get embeddings for a list of text IDs
- `GET /health` - Health check endpoint
- `GET /metrics` - Get server performance metrics
- `GET /` - Server information

## Performance Considerations

For optimal performance:

1. **Memory Usage**: The server loads all embeddings into memory for fast access. Ensure your server has sufficient RAM.

2. **Worker Count**: Adjust the number of workers based on your CPU cores and available memory:
   ```
   workers = min(8, (num_cores * 2) + 1)
   ```

3. **Batch Requests**: When possible, batch multiple text IDs in a single request.

4. **Concurrency**: The client supports concurrent requests for higher throughput.

5. **Monitoring**: Use the `/metrics` endpoint to monitor request rates and response times.

## Comparison with DynamoDB Implementation

This local embedding server is designed as a replacement for the DynamoDB-based implementation in `scripts/async_embedding_server.py` and `scripts/dynamo_utils.py`. Key differences:

1. **Storage**: Uses local memory instead of DynamoDB
2. **Performance**: Optimized for high-throughput local access
3. **Setup**: No AWS credentials or setup required
4. **Interface**: Provides a compatible API for easy migration

## Directory Structure

The embedding files are expected to be organized as follows:

```
/path/to/embedding/directory/
├── docs/
│   ├── 0.bin
│   ├── 1.bin
│   ├── ...
│   └── 7.bin
└── queries/
    ├── 0.bin
    ├── 1.bin
    ├── ...
    └── 7.bin
```

Each `.bin` file contains embedding data in the pickle format, where each object is a tuple of (vector_ids, vectors).
