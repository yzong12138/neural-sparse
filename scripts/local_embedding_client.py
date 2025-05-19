import os
import threading
import numpy as np
import logging
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    def __init__(self, server_url="http://localhost:8000", max_workers=10):
        """Initialize the local embedding service with server URL and worker threads"""
        self.server_url = server_url
        
        # Initialize internal state for async operations
        self.registered_tasks = {}
        self.fetched_embeddings = {}
        self.task_events = {}
        self.lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def fetch_embeddings_from_server(self, text_ids: List[int]):
        """Background task to fetch embeddings from local server"""
        try:
            key = f"{','.join(map(str, text_ids))}"
            response = requests.post(
                f"{self.server_url}/embeddings",
                json={"text_ids": text_ids}
            )
            response.raise_for_status()
            
            # Parse response and convert to numpy arrays
            data = response.json()
            raw_embeddings = data["embeddings"]
            embeddings = []
            for emb in raw_embeddings:
                if emb is not None:
                    embeddings.append(np.array(emb, dtype=np.float16))
                else:
                    embeddings.append(None)
            
            # Store results and set event flag
            with self.lock:
                self.fetched_embeddings[key] = np.array(embeddings)
                if key in self.task_events:  # Ensure task still exists
                    self.task_events[key].set()
        except Exception as e:
            # Log error and set event to avoid deadlocks
            logger.error(f"Error fetching embeddings: {e}")
            with self.lock:
                self.fetched_embeddings[key] = {"error": str(e)}
                if key in self.task_events:
                    self.task_events[key].set()

    def register_task(self, text_ids: List[int]) -> Dict[str, Any]:
        """Register a task to fetch embeddings"""
        key = f"{','.join(map(str, text_ids))}"

        with self.lock:
            # Update or create task
            self.registered_tasks[key] = self.registered_tasks.get(key, 0) + 1

            # If it's a new task, create event and submit to thread pool
            if key not in self.task_events:
                self.task_events[key] = Event()
                needs_submit = True
            else:
                needs_submit = False

        # Submit task outside the lock
        if needs_submit:
            self.thread_pool.submit(
                self.fetch_embeddings_from_server,
                text_ids,
            )

        return {"status": "success", "task_id": key}

    def fetch_embedding(self, text_ids: List[int]) -> np.ndarray:
        """Fetch embeddings, wait if task not completed"""
        key = f"{','.join(map(str, text_ids))}"

        # Check if task exists
        with self.lock:
            if key not in self.registered_tasks:
                # Register a new task if it doesn't exist
                self.register_task(text_ids)

            # If results are available, return directly
            if key in self.fetched_embeddings:
                embeddings = self.fetched_embeddings[key]
                event = None
            else:
                # Get event object to wait outside the lock
                event = self.task_events[key]
                embeddings = None

        # Wait for task completion outside the lock
        if event:
            event.wait()
            # Get results after waiting
            with self.lock:
                if key in self.fetched_embeddings:
                    embeddings = self.fetched_embeddings[key]
                else:
                    raise RuntimeError("Task result not available")

        # Decrease task count and clean up completed tasks
        with self.lock:
            self.registered_tasks[key] -= 1
            if self.registered_tasks[key] <= 0:
                # Clean up completed tasks
                self.registered_tasks.pop(key, None)
                self.fetched_embeddings.pop(key, None)
                self.task_events.pop(key, None)

        # Check if result contains error information
        if isinstance(embeddings, dict) and "error" in embeddings:
            raise RuntimeError(f"Task failed: {embeddings['error']}")

        return embeddings

    def health_check(self) -> Dict[str, str]:
        """Health check method"""
        try:
            response = requests.get(f"{self.server_url}/health")
            response.raise_for_status()
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def shutdown(self):
        """Clean up resources when shutting down"""
        self.thread_pool.shutdown(wait=True)


# Direct/synchronous client for simple use cases
class SimpleLocalEmbeddingClient:
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url

    def get_embeddings(self, text_ids: List[int]) -> np.ndarray:
        """Get embeddings for the given text IDs"""
        response = requests.post(
            f"{self.server_url}/embeddings",
            json={"text_ids": text_ids}
        )
        response.raise_for_status()
        
        data = response.json()
        raw_embeddings = data["embeddings"]
        embeddings = []
        for emb in raw_embeddings:
            if emb is not None:
                embeddings.append(np.array(emb, dtype=np.float16))
            else:
                embeddings.append(None)
        
        return np.array(embeddings)
