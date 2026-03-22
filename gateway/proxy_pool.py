"""
proxy_pool.py - 代理池管理器（含熔断器）
优化：新增 HTTP 健康检查探活，支持自动节点恢复
"""
import asyncio
import time
import httpx
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random


class CircuitState(Enum):
    CLOSED = "closed"        # 正常通行
    OPEN = "open"            # 熔断中
    HALF_OPEN = "half_open"  # 试探恢复


@dataclass
class ProxyNode:
    name: str
    url: str
    weight: int = 100
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    response_time_ms: float = 0.0
    total_requests: int = 0
    health_ok: bool = True  # 新增：HTTP 探活结果

    # 熔断阈值
    failure_threshold: int = 3
    recovery_timeout: float = 30.0


class ProxyPool:
    def __init__(self, nodes: list["ProxyNode"]):
        self.nodes = nodes
        self._lock = asyncio.Lock()

    def get_available(self) -> Optional["ProxyNode"]:
        """加权随机选取可用节点"""
        available = []
        for node in self.nodes:
            if node.state == CircuitState.CLOSED and node.health_ok:
                available.append((node, node.weight))
            elif node.state == CircuitState.HALF_OPEN:
                available.append((node, 1))

        if not available:
            # 全部熔断时，回退到最早失败的节点
            open_nodes = [n for n in self.nodes if n.state == CircuitState.OPEN]
            if open_nodes:
                return min(open_nodes, key=lambda n: n.last_failure_time)
            return None

        total = sum(w for _, w in available)
        r = random.uniform(0, total)
        cumulative = 0
        for node, weight in available:
            cumulative += weight
            if r <= cumulative:
                return node
        return available[-1][0]

    async def report_success(self, node: "ProxyNode", response_time_ms: float):
        async with self._lock:
            node.success_count += 1
            node.failure_count = 0
            node.last_success_time = time.time()
            node.response_time_ms = response_time_ms
            node.total_requests += 1
            node.health_ok = True
            if node.state == CircuitState.HALF_OPEN:
                node.state = CircuitState.CLOSED
                print(f"[Circuit] {node.name}: HALF_OPEN → CLOSED ✅")

    async def report_failure(self, node: "ProxyNode", error: str):
        async with self._lock:
            node.failure_count += 1
            node.last_failure_time = time.time()
            node.total_requests += 1
            if (node.state == CircuitState.CLOSED
                    and node.failure_count >= node.failure_threshold):
                node.state = CircuitState.OPEN
                print(f"[Circuit] {node.name}: CLOSED → OPEN ❌ ({error})")
            elif node.state == CircuitState.HALF_OPEN:
                node.state = CircuitState.OPEN
                print(f"[Circuit] {node.name}: HALF_OPEN → OPEN ❌")

    async def maybe_recover(self):
        """检查是否可将 OPEN 节点恢复为 HALF_OPEN"""
        now = time.time()
        for node in self.nodes:
            if (node.state == CircuitState.OPEN
                    and now - node.last_failure_time > node.recovery_timeout):
                node.state = CircuitState.HALF_OPEN
                node.failure_count = 0
                print(f"[Circuit] {node.name}: OPEN → HALF_OPEN 🔄")

    async def health_check_all(self):
        """主动 HTTP 探活所有节点（优化新增）"""
        async def check_one(node: "ProxyNode"):
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.get(f"{node.url}/health")
                    node.health_ok = resp.status_code == 200
            except Exception:
                node.health_ok = False
                if node.state == CircuitState.CLOSED:
                    node.state = CircuitState.OPEN
                    node.last_failure_time = time.time()
                    print(f"[HealthCheck] {node.name}: 探活失败 → OPEN ⚠️")

        await asyncio.gather(*[check_one(n) for n in self.nodes])

    def status(self) -> dict:
        return {
            node.name: {
                "state": node.state.value,
                "health_ok": node.health_ok,
                "failures": node.failure_count,
                "successes": node.success_count,
                "avg_response_ms": round(node.response_time_ms, 1),
                "total_requests": node.total_requests,
                "url": node.url,
            }
            for node in self.nodes
        }
