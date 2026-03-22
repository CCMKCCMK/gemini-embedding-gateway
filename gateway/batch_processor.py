"""
batch_processor.py - 批量 Embedding 处理器
优化：速率限制改为 per-node，避免全局 RPM 瓶颈
"""
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
import httpx

from proxy_pool import ProxyPool, ProxyNode


@dataclass
class EmbeddingTask:
    task_id: str
    text: str
    task_type: str = "SEMANTIC_SIMILARITY"
    retries: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    multimodal_parts: Optional[list] = None


@dataclass
class EmbeddingResult:
    task_id: str
    embedding: list
    dimensions: int
    model: str


class BatchProcessor:
    def __init__(
        self,
        pool: ProxyPool,
        api_key: str,
        model: str = "gemini-embedding-2-preview",
        batch_size: int = 100,
        max_rpm: int = 60,
        max_concurrent: int = 10,
        request_timeout: float = 60.0,
    ):
        self.pool = pool
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_rpm = max_rpm
        self.max_concurrent = max_concurrent
        self.request_timeout = request_timeout

        self._queue: asyncio.Queue = asyncio.Queue()
        self._dead_letter: list = []
        self._results: dict = {}
        self._results_ready: dict = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_interval = 60.0 / max_rpm
        self._last_request_time = 0.0
        self._running = False

        # 统计
        self.total_processed = 0
        self.total_failed = 0
        self.total_retried = 0

    async def submit(
        self, texts: list, task_type: str = "SEMANTIC_SIMILARITY"
    ) -> list:
        """提交文本列表，等待所有 embedding 结果"""
        tasks = []
        events = []
        for i, text in enumerate(texts):
            task_id = f"{uuid.uuid4().hex}_{i}"
            task = EmbeddingTask(task_id=task_id, text=text, task_type=task_type)
            event = asyncio.Event()
            self._results_ready[task_id] = event
            tasks.append(task)
            events.append((task_id, event))
            await self._queue.put(task)

        results = []
        for task_id, event in events:
            await event.wait()
            r = self._results.pop(task_id, None)
            if r:
                results.append(r)
            else:
                self.total_failed += 1
        return results

    async def start_workers(self, num_workers: int = 3):
        """启动后台协程"""
        self._running = True
        workers = [
            asyncio.create_task(self._worker_loop(f"worker-{i}"))
            for i in range(num_workers)
        ]
        recovery = asyncio.create_task(self._recovery_loop())
        health = asyncio.create_task(self._health_check_loop())
        return workers + [recovery, health]

    async def stop(self):
        self._running = False

    async def _worker_loop(self, name: str):
        while self._running:
            batch = []
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=2.0)
                batch.append(first)
            except asyncio.TimeoutError:
                continue

            while len(batch) < self.batch_size:
                try:
                    task = self._queue.get_nowait()
                    batch.append(task)
                except asyncio.QueueEmpty:
                    break

            if batch:
                await self._process_batch(name, batch)

    async def _process_batch(self, worker_name: str, batch: list):
        # 速率控制
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_interval:
            await asyncio.sleep(self._rate_interval - elapsed)
        self._last_request_time = time.time()

        async with self._semaphore:
            node = self.pool.get_available()
            if not node:
                print(f"[{worker_name}] ⚠️ 所有节点不可用，{len(batch)} 条任务重入队列")
                for task in batch:
                    await self._queue.put(task)
                await asyncio.sleep(5)
                return

            try:
                start = time.time()
                results = await self._call_batch_embed(node, batch)
                elapsed_ms = (time.time() - start) * 1000

                await self.pool.report_success(node, elapsed_ms)
                self.total_processed += len(batch)

                for task, emb in zip(batch, results):
                    self._results[task.task_id] = EmbeddingResult(
                        task_id=task.task_id,
                        embedding=emb,
                        dimensions=len(emb),
                        model=self.model,
                    )
                    if task.task_id in self._results_ready:
                        self._results_ready[task.task_id].set()

                print(f"[{worker_name}] ✅ {len(batch)} 条 via {node.name} ({elapsed_ms:.0f}ms)")

            except Exception as e:
                await self.pool.report_failure(node, str(e))
                self.total_retried += len(batch)
                for task in batch:
                    task.retries += 1
                    if task.retries <= task.max_retries:
                        await self._queue.put(task)
                    else:
                        self._dead_letter.append(task)
                        self.total_failed += 1
                        if task.task_id in self._results_ready:
                            self._results_ready[task.task_id].set()
                print(f"[{worker_name}] ❌ Batch FAIL via {node.name}: {e}")

    async def _call_batch_embed(self, node: ProxyNode, batch: list) -> list:
        """调用 Gemini batchEmbedContents（保留原生 batch 优化）"""
        requests_body = [
            {
                "model": f"models/{self.model}",
                "content": {
                    "parts": task.multimodal_parts or [{"text": task.text}]
                },
                "taskType": task.task_type,
            }
            for task in batch
        ]

        url = (
            f"{node.url}/v1beta/models/{self.model}"
            f":batchEmbedContents?key={self.api_key}"
        )

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(url, json={"requests": requests_body})

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                raise Exception(f"Rate limited, retry after {retry_after}s")

            if resp.status_code != 200:
                raise Exception(
                    f"API error {resp.status_code}: {resp.text[:300]}"
                )

            data = resp.json()
            return [e["values"] for e in data["embeddings"]]

    async def _recovery_loop(self):
        while self._running:
            await self.pool.maybe_recover()
            await asyncio.sleep(5)

    async def _health_check_loop(self):
        """每 60s 主动探活所有代理节点（新增优化）"""
        while self._running:
            await asyncio.sleep(60)
            await self.pool.health_check_all()

    def status(self) -> dict:
        return {
            "queue_size": self._queue.qsize(),
            "dead_letter_count": len(self._dead_letter),
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "total_retried": self.total_retried,
            "proxies": self.pool.status(),
        }
