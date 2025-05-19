import os
import numpy as np
import pickle
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any
import uvicorn
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("embedding-server")

app = FastAPI(title="Local Embedding Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmbeddingRequest(BaseModel):
    text_ids: List[int]
    model_id: int

class EmbeddingStore:
    def __init__(self):
        # Store embeddings per model_id
        self.model_embeddings: Dict[int, Dict[int, np.ndarray]] = {}
        self.model_dimensions: Dict[int, int] = {}
        # Performance metrics
        self.request_count = 0
        self.total_processing_time = 0
        self.start_time = datetime.now()

    def load_embeddings(self, directory: str, model_id: int):
        """
        Load embeddings from binary files into memory for a specific model_id
        """
        start_time = time.time()
        total_embeddings = 0
        
        # Initialize model embeddings dictionary if not exists
        if model_id not in self.model_embeddings:
            self.model_embeddings[model_id] = {}
        
        try:
            for subdir in ['queries', 'docs']:
                subdir_path = os.path.join(directory, subdir)
                if not os.path.exists(subdir_path):
                    logger.warning(f"Directory not found: {subdir_path}")
                    continue
                    
                for i in range(8):
                    file_path = os.path.join(subdir_path, f"{i}.bin")
                    if not os.path.exists(file_path):
                        logger.warning(f"File not found: {file_path}")
                        continue
                        
                    logger.info(f"Loading embeddings from {file_path} for model {model_id}")
                    batch_count = 0
                    with open(file_path, "rb") as f:
                        while True:
                            try:
                                data = pickle.load(f)
                                vector_ids, vectors = data
                                
                                # Set embedding dimension from first loaded vector
                                if model_id not in self.model_dimensions and len(vectors) > 0:
                                    self.model_dimensions[model_id] = vectors[0].shape[0]
                                    logger.info(f"Model {model_id} embedding dimension: {self.model_dimensions[model_id]}")
                                
                                # Add vectors to the dictionary
                                for id, vector in zip(vector_ids, vectors):
                                    self.model_embeddings[model_id][id] = vector
                                
                                batch_count += 1
                                total_embeddings += len(vector_ids)
                            except EOFError:
                                break
                    
                    logger.info(f"Loaded {batch_count} batches from {file_path}")
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {total_embeddings} embeddings for model {model_id} in {load_time:.2f}s ({total_embeddings/load_time if load_time > 0 else 0:.2f} embeddings/s)")
            logger.info(f"Total unique embeddings for model {model_id}: {len(self.model_embeddings[model_id])}")
            
        except Exception as e:
            logger.error(f"Error loading embeddings for model {model_id}: {str(e)}")
            raise

    def get_embeddings(self, text_ids: List[int], model_id: int) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for the given text IDs from a specific model
        Returns a list of embeddings, with None for IDs that don't exist
        """
        # Check if model exists
        if model_id not in self.model_embeddings:
            logger.warning(f"Model {model_id} not found in embedding store")
            return [None] * len(text_ids)
            
        # Optimize lookup for better performance
        result = []
        model_embeddings = self.model_embeddings[model_id]
        for id in text_ids:
            emb = model_embeddings.get(id)
            result.append(emb)
        return result

# Global store instance
embedding_store = EmbeddingStore()

# Middleware to track request metrics
@app.middleware("http")
async def track_request_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Only track /embeddings endpoints for performance metrics
    if request.url.path == "/embeddings":
        embedding_store.request_count += 1
        embedding_store.total_processing_time += process_time
    
    return response

@app.post("/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    """
    Get embeddings for the given text IDs and model ID
    """
    if not request.text_ids:
        return {"embeddings": []}
    
    start_time = time.time()
    embeddings = embedding_store.get_embeddings(request.text_ids, request.model_id)
    
    # Convert to list for JSON serialization
    result = [emb.tolist() if emb is not None else None for emb in embeddings]
    
    elapsed = time.time() - start_time
    if elapsed > 0.1:  # Log slow requests
        logger.warning(f"Slow request: {len(request.text_ids)} embeddings for model {request.model_id} in {elapsed:.4f}s")
    
    return {"embeddings": result}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "embeddings_loaded": len(embedding_store.embeddings),
        "embedding_dimension": embedding_store.embedding_dimension,
        "uptime_seconds": (datetime.now() - embedding_store.start_time).total_seconds()
    }

@app.get("/metrics")
async def get_metrics():
    """
    Get server performance metrics
    """
    uptime_seconds = (datetime.now() - embedding_store.start_time).total_seconds()
    
    return {
        "request_count": embedding_store.request_count,
        "uptime_seconds": uptime_seconds,
        "requests_per_second": embedding_store.request_count / uptime_seconds if uptime_seconds > 0 else 0,
        "avg_processing_time": embedding_store.total_processing_time / embedding_store.request_count 
                              if embedding_store.request_count > 0 else 0,
        "embeddings_count": len(embedding_store.embeddings),
        "embedding_dimension": embedding_store.embedding_dimension
    }

@app.get("/")
async def root():
    """
    Root endpoint with server information
    """
    return {
        "name": "Local Embedding Server",
        "version": "1.0.0",
        "description": "Serves embeddings from local files",
        "endpoints": [
            {"path": "/embeddings", "method": "POST", "description": "Get embeddings for text IDs"},
            {"path": "/health", "method": "GET", "description": "Server health check"},
            {"path": "/metrics", "method": "GET", "description": "Server performance metrics"},
        ]
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start the local embedding server")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing embedding files")
    parser.add_argument("--model_id", type=int, required=True, help="Model ID")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--workers", type=int, default=4, help="Number of uvicorn workers")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    args = parser.parse_args()

    # Load embeddings before starting the server
    logger.info(f"Loading embeddings from {args.directory} for model {args.model_id}")
    embedding_store.load_embeddings(args.directory, args.model_id)
    
    # Configure server options for optimal performance
    logger.info(f"Starting server on port {args.port} with {args.workers} workers")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=args.port,
        workers=args.workers,
        log_level=args.log_level
    )
