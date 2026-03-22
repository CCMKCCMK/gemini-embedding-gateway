"""
gateway.py - 统一网关入口
修复原版 lifespan 重复定义 bug，新增监控 Dashboard
"""
import asyncio
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from proxy_pool import ProxyPool, ProxyNode
from batch_processor import BatchProcessor

# ===================== 配置读取 =====================
def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

CONFIG = {
    "google_api_key": _env("GOOGLE_API_KEY"),
    "auth_token":     _env("GATEWAY_TOKEN", "sk-embed-gateway"),
    "model":          _env("MODEL", "gemini-embedding-2-preview"),
    "batch_size":     _env_int("BATCH_SIZE", 100),
    "max_rpm":        _env_int("MAX_RPM", 60),
    "max_concurrent": _env_int("MAX_CONCURRENT", 10),
    "proxies": [
        {"name": "proxy-1", "url": _env("PROXY_1", ""), "weight": 100},
        {"name": "proxy-2", "url": _env("PROXY_2", ""), "weight": 80},
        {"name": "proxy-3", "url": _env("PROXY_3", ""), "weight": 60},
    ],
}

# 过滤掉未配置的节点
active_proxies = [p for p in CONFIG["proxies"] if p["url"].startswith("http")]

if not active_proxies:
    raise RuntimeError(
        "❌ 未检测到有效的 PROXY_1/PROXY_2/PROXY_3 环境变量！请在 .env 中填写代理节点地址。"
    )

if not CONFIG["google_api_key"]:
    raise RuntimeError("❌ 未检测到 GOOGLE_API_KEY 环境变量！")

nodes = [ProxyNode(name=p["name"], url=p["url"].rstrip("/"), weight=p["weight"])
         for p in active_proxies]
pool = ProxyPool(nodes)

processor = BatchProcessor(
    pool=pool,
    api_key=CONFIG["google_api_key"],
    model=CONFIG["model"],
    batch_size=CONFIG["batch_size"],
    max_rpm=CONFIG["max_rpm"],
    max_concurrent=CONFIG["max_concurrent"],
)

# ===================== FastAPI =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时主动探活
    await pool.health_check_all()
    workers = await processor.start_workers(num_workers=5)
    print(f"🚀 Gateway 已启动，{len(nodes)} 个代理节点在线")
    yield
    await processor.stop()
    for w in workers:
        w.cancel()

app = FastAPI(
    title="Gemini Embedding Gateway",
    description="国内高可用 Gemini Embedding 2 接入网关",
    version="2.0.0",
    lifespan=lifespan,
)

# ===================== 鉴权 =====================
async def verify_token(authorization: str = Header(None)):
    token = (authorization or "").replace("Bearer ", "").strip()
    if token != CONFIG["auth_token"]:
        raise HTTPException(status_code=401, detail="无效的 API Key")

# ===================== 请求模型 =====================
class EmbedRequest(BaseModel):
    input: str | list
    model: str = "gemini-embedding-2-preview"
    task_type: str = "SEMANTIC_SIMILARITY"

class BatchEmbedRequest(BaseModel):
    texts: list
    task_type: str = "SEMANTIC_SIMILARITY"

# ===================== API 端点 =====================

@app.post("/v1/embeddings", dependencies=[Depends(verify_token)])
async def openai_embeddings(req: EmbedRequest):
    """OpenAI 兼容 /v1/embeddings 端点"""
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    results = await processor.submit(texts, req.task_type)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": r.embedding, "index": i}
            for i, r in enumerate(results)
        ],
        "model": req.model,
        "usage": {
            "prompt_tokens": sum(len(str(t).split()) for t in texts),
            "total_tokens": sum(len(str(t).split()) for t in texts),
        },
    }


@app.post("/v1/batch-embed", dependencies=[Depends(verify_token)])
async def batch_embed(req: BatchEmbedRequest):
    """批量提交端点，适合大规模 RAG 建索引"""
    results = await processor.submit(req.texts, req.task_type)
    return {
        "count": len(results),
        "embeddings": [
            {"task_id": r.task_id, "embedding": r.embedding, "dimensions": r.dimensions}
            for r in results
        ],
    }


@app.post("/v1beta/models/{model_path:path}", dependencies=[Depends(verify_token)])
async def native_proxy(model_path: str, request: Request):
    """直通 Google 原生 API（保留兼容性）"""
    node = pool.get_available()
    if not node:
        raise HTTPException(status_code=503, detail="所有代理节点不可用")

    body = await request.json()
    url = f"{node.url}/v1beta/models/{model_path}?key={CONFIG['google_api_key']}"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=body)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        await pool.report_failure(node, str(e))
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/health")
async def health():
    """健康检查"""
    status = processor.status()
    all_ok = any(v.get("health_ok") for v in status.get("proxies", {}).values())
    return {
        "status": "ok" if all_ok else "degraded",
        "timestamp": time.time(),
        **status,
    }


@app.get("/metrics")
async def metrics():
    """指标端点"""
    return processor.status()


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """可视化监控 Dashboard（新增优化）"""
    status = processor.status()
    proxies = status.get("proxies", {})

    rows = ""
    for name, info in proxies.items():
        state = info.get("state", "unknown")
        health = "✅" if info.get("health_ok") else "❌"
        color = {"closed": "#28a745", "open": "#dc3545", "half_open": "#ffc107"}.get(state, "#6c757d")
        rows += f"""
        <tr>
          <td><code>{name}</code></td>
          <td><a href="{info['url']}" target="_blank">{info['url']}</a></td>
          <td style="color:{color};font-weight:bold">{state}</td>
          <td>{health}</td>
          <td>{info.get('avg_response_ms', 0):.0f} ms</td>
          <td>{info.get('total_requests', 0)}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="10">
  <title>Gemini Embedding Gateway</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; margin: 40px; background: #f8f9fa; }}
    h1 {{ color: #343a40; }}
    .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ background: #343a40; color: white; padding: 10px; text-align: left; }}
    td {{ padding: 10px; border-bottom: 1px solid #dee2e6; }}
    .stat {{ display: inline-block; margin: 10px 20px 10px 0; }}
    .stat-num {{ font-size: 2em; font-weight: bold; color: #007bff; }}
    .stat-label {{ color: #6c757d; font-size: 0.9em; }}
    .badge {{ padding: 3px 8px; border-radius: 4px; color: white; background: #28a745; }}
  </style>
</head>
<body>
  <h1>🛡️ Gemini Embedding Gateway <span class="badge">v2.0</span></h1>
  <div class="card">
    <h3>📊 处理统计</h3>
    <div class="stat"><div class="stat-num">{status.get('total_processed', 0)}</div><div class="stat-label">已处理</div></div>
    <div class="stat"><div class="stat-num">{status.get('queue_size', 0)}</div><div class="stat-label">队列中</div></div>
    <div class="stat"><div class="stat-num">{status.get('total_failed', 0)}</div><div class="stat-label">失败</div></div>
    <div class="stat"><div class="stat-num">{status.get('dead_letter_count', 0)}</div><div class="stat-label">死信</div></div>
  </div>
  <div class="card">
    <h3>🌐 代理节点状态</h3>
    <table>
      <tr><th>节点名</th><th>地址</th><th>熔断状态</th><th>探活</th><th>平均延迟</th><th>总请求</th></tr>
      {rows}
    </table>
  </div>
  <p style="color:#aaa;font-size:0.8em">每 10 秒自动刷新 · 模型: {CONFIG['model']} · RPM: {CONFIG['max_rpm']}</p>
</body>
</html>"""
