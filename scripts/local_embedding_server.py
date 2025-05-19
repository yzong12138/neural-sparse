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
        result = []
        
        # Check if model_id exists
        if model_id not in self.model_embeddings:
            logger.warning(f"Model ID {model_id} not found in embedding store")
            return [None] * len(text_ids)
            
        model_embeddings = self.model_embeddings[model_id]
        for id in text_ids:
            emb = model_embeddings.get(id)
            result.append(emb)
        return result
        
    def get_available_models(self) -> Dict[int, int]:
        """
        Returns a dictionary of available model IDs and their embedding dimensions
        """
        return {model_id: dim for model_id, dim in self.model_dimensions.items()}

# Global store instance
embedding_store = EmbeddingStore()

@app.post("/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    """
    Get embeddings for the given text IDs and model ID
    """
    embeddings = embedding_store.get_embeddings(request.text_ids, request.model_id)
    result = [emb.tolist() if emb is not None else None for emb in embeddings]
    
    return {"embeddings": result}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy"
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start the local embedding server")
    parser.add_argument("--directories", type=str, nargs="+", required=True, help="Directories containing embedding files")
    parser.add_argument("--model_ids", type=int, nargs="+", required=True, help="Model IDs corresponding to directories")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    args = parser.parse_args()

    # Validate that directories and model_ids have the same length
    if len(args.directories) != len(args.model_ids):
        raise ValueError("Number of directories must match number of model IDs")

    # Load embeddings before starting the server
    for directory, model_id in zip(args.directories, args.model_ids):
        logger.info(f"Loading embeddings from {directory} for model {model_id}")
        embedding_store.load_embeddings(directory, model_id)
    
    # Configure server options for optimal performance
    logger.info(f"Starting server on port {args.port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=args.port,
        log_level=args.log_level
    )

# python scripts/local_embedding_server.py --directories /opt/dlami/nvme/bge-256 /opt/dlami/nvme/inf-256 --model_id 4 5
# python scripts/local_embedding_server.py --directories /opt/dlami/nvme/inf-256 --model_id 5