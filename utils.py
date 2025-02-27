#utils.py
import asyncio
import time
from collections import deque
import aiofiles
import json
from typing import Dict, List, Any, Optional,Callable,TypeVar

T = TypeVar('T')

class RateLimiter:
    def __init__(self, requests_per_second: int):
        self.rate = requests_per_second
        self.request_times = deque(maxlen=requests_per_second)
        self.lock = asyncio.Lock()
        self.successful_requests = deque(maxlen=30 * requests_per_second)
        self.last_display_time = time.time()

    async def record_success(self):
        async with self.lock:
            current_time = time.time()
            self.successful_requests.append(current_time)
            if current_time - self.last_display_time >= 1:
                self.display_current_rps()
                self.last_display_time = current_time

    def display_current_rps(self):
        current_time = time.time()
        while self.successful_requests and current_time - self.successful_requests[0] > 30:
            self.successful_requests.popleft()
        actual_rps = len(self.successful_requests) / 30 if self.successful_requests else 0
        print(f"\rActual RPS (last 30s): {actual_rps:.2f}", end="", flush=True)

    async def acquire(self):
        async with self.lock:
            now = time.time()
            while self.request_times and now - self.request_times[0] >= 1:
                self.request_times.popleft()
            if len(self.request_times) < self.rate:
                self.request_times.append(now)
                return
            wait_time = 1 - (now - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.request_times.append(time.time())

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

async def process_with_retry(func: Callable[..., T], *args: Any, max_retries: int = 3, base_delay: int = 3) -> T:
    for attempt in range(max_retries):
        try:
            return await func(*args)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
            await asyncio.sleep(delay)


async def append_result_jsonl(result: Dict, output_path: str):
    try:
        async with aiofiles.open(output_path, 'a', encoding='utf-8') as f:
            # Convert result to a JSON-serializable format
            serializable_result = {
                "pdf_name": result["pdf_name"],
                "page_number": result["page_number"],
                "queries": result["queries"],
                "error": result["error"],
                "processing_time": result["processing_time"]
            }
            await f.write(json.dumps(serializable_result, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error writing to output file: {str(e)}")