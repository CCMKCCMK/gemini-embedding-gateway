"""
test.py - 快速验证脚本（仅用标准库，无需安装依赖）
运行方式：python scripts/test.py
"""
import sys
import os
import json
import urllib.request

# 读取 .env 文件
env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

GATEWAY_URL   = os.getenv("GATEWAY_URL",   "http://localhost:8000")
GATEWAY_TOKEN = os.getenv("GATEWAY_TOKEN", "sk-embed-gateway")

print(f"🔍 测试网关: {GATEWAY_URL}")
print(f"🔑 使用 Token: {GATEWAY_TOKEN[:12]}...")

try:
    # ── 健康检查 ──
    req_health = urllib.request.Request(f"{GATEWAY_URL}/health")
    with urllib.request.urlopen(req_health, timeout=10) as resp:
        health = json.loads(resp.read())
    print(f"✅ 健康检查通过：状态 = {health.get('status')}")

    # ── Embedding 测试 ──
    body = json.dumps({
        "input": ["测试文本1", "测试文本2", "Hello World"]
    }).encode("utf-8")
    req_embed = urllib.request.Request(
        f"{GATEWAY_URL}/v1/embeddings",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GATEWAY_TOKEN}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req_embed, timeout=30) as resp:
        data = json.loads(resp.read())

    print(f"✅ Embedding 测试通过！")
    print(f"   返回数量 : {len(data['data'])} 个")
    print(f"   向量维度 : {len(data['data'][0]['embedding'])}")
    print(f"   使用模型 : {data.get('model', 'N/A')}")
    sys.exit(0)

except urllib.error.HTTPError as e:
    print(f"❌ HTTP 错误 {e.code}: {e.reason}")
    if e.code == 401:
        print("   → GATEWAY_TOKEN 不正确，请检查 .env 文件")
    elif e.code == 503:
        print("   → 所有代理节点不可用，请检查 PROXY_1/2/3 是否正确")
    sys.exit(1)
except Exception as e:
    print(f"❌ 连接失败: {e}")
    print("\n请检查：")
    print("  1. Docker Desktop 是否运行，网关容器是否 Running")
    print("  2. 访问 http://localhost:8000 是否能打开监控页面")
    sys.exit(1)
