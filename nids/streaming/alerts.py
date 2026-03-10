"""
Publish/Subscribe alert pipeline for the SOC Dashboard.
Uses an in-memory asyncio Queue by default for local testing,
with standard interfaces that can be extended for Redis/Kafka if needed.
"""

import asyncio
import json
import logging
from typing import Dict, Any, AsyncGenerator

logger = logging.getLogger("nids.streaming")

class AlertProvider:
    def __init__(self):
        # By default we use an in-memory queue. 
        # In a real distributed system, this would be a Redis connection (e.g. using aioredis)
        # We will keep a set of queues for active websocket connections.
        self.subscribers = set()
        
    def add_subscriber(self, queue: asyncio.Queue):
        self.subscribers.add(queue)
        
    def remove_subscriber(self, queue: asyncio.Queue):
        if queue in self.subscribers:
            self.subscribers.remove(queue)
            
    async def publish(self, alert_data: Dict[str, Any]):
        """Publish an alert to all active subscribers."""
        if not self.subscribers:
            return
            
        payload = json.dumps(alert_data)
        for queue in self.subscribers:
            # Non-blocking put, we won't wait if a queue is somehow full, 
            # though default queues are infinite.
            queue.put_nowait(payload)
            
# Global instance for the FastAPI application to share memory across requests
# (Assuming single-process uvicorn for this phase. If using gunicorn with multiple workers,
# we would need Redis).
alert_provider = AlertProvider()

class AlertSubscriber:
    """Async iterator that yields events from the provider."""
    def __init__(self):
        self.queue = asyncio.Queue()

    async def __aenter__(self):
        alert_provider.add_subscriber(self.queue)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        alert_provider.remove_subscriber(self.queue)

    async def __aiter__(self):
        return self
        
    async def __anext__(self) -> str:
        """Wait for the next message from the queue."""
        payload = await self.queue.get()
        self.queue.task_done()
        return payload
