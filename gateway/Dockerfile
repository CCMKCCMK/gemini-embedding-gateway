# ============================================================
# Gateway Dockerfile
# 本地通过 Docker Desktop 一键启动
# ============================================================
FROM python:3.12-slim

WORKDIR /app

# 安装依赖（独立层，利用 Docker 缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8000"]
