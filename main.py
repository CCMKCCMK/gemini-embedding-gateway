"""
Gemini Embedding Gateway - Railway 直连版
无需 nginx 代理，直接从海外服务器调用 Google Gemini API
支持 OpenAI 兼容 /v1/embeddings 接口 + batchEmbedContents
"""
import asyncio
import os
import time
import logging
from typing import List, Union, Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gemini-gateway")

# ─── 配置 ────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
GATEWAY_TOKEN   = os.getenv("GATEWAY_TOKEN", "sk-gemini-gateway-2025")
MODEL           = os.getenv("MODEL", "gemini-embedding-2-preview")
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", "100"))
MAX_RPM         = int(os.getenv("MAX_RPM", "60"))
PORT            = int(os.getenv("PORT", "8000"))

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com"
    f"/v1beta/models/{MODEL}:batchEmbedContents"
    f"?key={GOOGLE_API_KEY}"
)

# ─── 速率限制（简单令牌桶）───────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()

limiter = RateLimiter(MAX_RPM)

# ─── 请求/响应模型 ────────────────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    task_type: Optional[str] = "RETRIEVAL_DOCUMENT"  # 扩展字段

# ─── 核心：调用 Google batchEmbedContents ──────────────────────────────────
async def batch_embed(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """把文本列表拆成每批 ≤BATCH_SIZE 条，逐批调用 Google API，返回向量列表"""
    if not GOOGLE_API_KEY:
        raise HTTPException(500, "GOOGLE_API_KEY 未配置")

    all_vectors: List[List[float]] = []

    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = texts[i : i + BATCH_SIZE]
            await limiter.acquire()

            payload = {
                "requests": [
                    {
                        "model": f"models/{MODEL}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task_type,
                    }
                    for t in chunk
                ]
            }

            for attempt in range(4):
                try:
                    resp = await client.post(GEMINI_URL, json=payload)
                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        logger.warning(f"429 速率限制，{wait}s 后重试（第{attempt+1}次）")
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    for emb in data.get("embeddings", []):
                        all_vectors.append(emb["values"])
                    break
                except httpx.HTTPStatusError as e:
                    if attempt == 3:
                        logger.error(f"Google API 错误: {e.response.text}")
                        raise HTTPException(502, f"Google API 错误: {e.response.status_code}")
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    if attempt == 3:
                        raise HTTPException(502, f"请求失败: {str(e)}")
                    await asyncio.sleep(2 ** attempt)

    return all_vectors

# ─── FastAPI ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Gemini Embedding Gateway 启动")
    logger.info(f"   模型: {MODEL}  批大小: {BATCH_SIZE}  限速: {MAX_RPM} RPM")
    logger.info(f"   API Key: {'✅ 已配置' if GOOGLE_API_KEY else '❌ 未配置！'}")
    yield
    logger.info("🛑 Gateway 关闭")

app = FastAPI(
    title="Gemini Embedding Gateway",
    description="OpenAI 兼容 Gemini Embedding 2 接入网关（Railway 直连版）",
    version="2.0.0",
    lifespan=lifespan,
)

# ─── 鉴权 ────────────────────────────────────────────────────────────────────
async def verify_token(authorization: str = Header(default="")):
    if not GATEWAY_TOKEN:
        return  # 未设置则不鉴权
    token = authorization.replace("Bearer ", "").strip()
    if token != GATEWAY_TOKEN:
        raise HTTPException(401, "Invalid token")

# ─── 接口 ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    ok = bool(GOOGLE_API_KEY)
    return JSONResponse(
        {"status": "ok" if ok else "degraded", "model": MODEL, "api_key_set": ok},
        status_code=200 if ok else 503,
    )

@app.post("/v1/embeddings")
async def embeddings(req: EmbedRequest, authorization: str = Header(default="")):
    await verify_token(authorization)
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(400, "input 不能为空")

    vectors = await batch_embed(texts, req.task_type or "RETRIEVAL_DOCUMENT")

    return {
        "object": "list",
        "model": req.model or MODEL,
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)},
    }

@app.post("/v1/batch-embed")
async def batch_embed_endpoint(req: EmbedRequest, authorization: str = Header(default="")):
    """别名，与旧版 gateway 兼容"""
    return await embeddings(req, authorization)

@app.get("/metrics")
async def metrics():
    return {
        "model": MODEL,
        "batch_size": BATCH_SIZE,
        "max_rpm": MAX_RPM,
        "api_key_set": bool(GOOGLE_API_KEY),
        "mode": "direct",
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    status_color = "#22c55e" if GOOGLE_API_KEY else "#ef4444"
    status_text  = "✅ 运行中" if GOOGLE_API_KEY else "❌ 缺少 API Key"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Gemini Embedding Gateway</title>
<style>
  body{{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:40px}}
  .card{{background:#1e293b;border-radius:12px;padding:28px;margin:16px 0;max-width:600px}}
  h1{{color:#38bdf8;margin-bottom:4px}}
  .badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:600}}
  .ok{{background:#166534;color:#86efac}} .err{{background:#7f1d1d;color:#fca5a5}}
  code{{background:#0f172a;padding:3px 8px;border-radius:4px;font-size:13px;color:#7dd3fc}}
  table{{width:100%;border-collapse:collapse}}
  td{{padding:8px 4px;border-bottom:1px solid #334155;font-size:14px}}
  td:first-child{{color:#94a3b8;width:140px}}
</style>
</head><body>
<h1>🌐 Gemini Embedding Gateway</h1>
<div class="card">
  <span class="badge {'ok' if GOOGLE_API_KEY else 'err'}">{status_text}</span>
  <table style="margin-top:16px">
    <tr><td>模型</td><td><code>{MODEL}</code></td></tr>
    <tr><td>批大小</td><td><code>{BATCH_SIZE} 条/批</code></td></tr>
    <tr><td>限速</td><td><code>{MAX_RPM} RPM</code></td></tr>
    <tr><td>模式</td><td><code>直连 Google API（无代理）</code></td></tr>
  </table>
</div>
<div class="card">
  <b>API 端点</b>
  <table style="margin-top:12px">
    <tr><td>嵌入接口</td><td><code>POST /v1/embeddings</code></td></tr>
    <tr><td>健康检查</td><td><code>GET /health</code></td></tr>
    <tr><td>监控指标</td><td><code>GET /metrics</code></td></tr>
  </table>
</div>
</body></html>"""
