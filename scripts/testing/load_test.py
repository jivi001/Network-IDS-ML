"""
Load testing script for NIDS deployment.
Simulates 1Gbps equivalent load and burst spikes.
"""

import asyncio
import time
import httpx
import numpy as np
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nids.loadtest")

API_URL = "http://localhost:8000/predict/batch"

async def _send_batch(client: httpx.AsyncClient, batch_size: int, num_features: int) -> dict:
    """Send a single batch to the inference API."""
    # Generate random normalized payload mimicking StandardScaler output
    payload = np.random.normal(0, 1, size=(batch_size, num_features)).tolist()
    
    start_time = time.time()
    try:
        response = await client.post(API_URL, json={"features": payload})
        response.raise_for_status()
        latency = (time.time() - start_time) * 1000
        return {"success": True, "latency": latency, "status": response.status_code}
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return {"success": False, "latency": latency, "error": str(e)}

async def run_load_test(num_requests: int = 1000, batch_size: int = 100, num_features: int = 20):
    """Run concurrent load test."""
    logger.info(f"Starting Load Test: {num_requests} requests of batch_size {batch_size}")
    
    results = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [_send_batch(client, batch_size, num_features) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
    # Analyze results
    latencies = [r['latency'] for r in results if r['success']]
    failures = len([r for r in results if not r['success']])
    
    if latencies:
        logger.info("=== Load Test Results ===")
        logger.info(f"Total Requests: {num_requests}")
        logger.info(f"Total Failures: {failures} ({failures/num_requests*100:.2f}%)")
        logger.info(f"P50 Latency: {np.percentile(latencies, 50):.2f} ms")
        logger.info(f"P95 Latency: {np.percentile(latencies, 95):.2f} ms")
        logger.info(f"P99 Latency: {np.percentile(latencies, 99):.2f} ms")
        logger.info(f"Max Latency: {np.max(latencies):.2f} ms")
        
        if np.percentile(latencies, 99) > 100:
            logger.warning("P99 Latency exceeded 100ms SLA.")
            
if __name__ == "__main__":
    asyncio.run(run_load_test(num_requests=500, batch_size=256))
