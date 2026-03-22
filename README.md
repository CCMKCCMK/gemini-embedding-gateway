# Gemini Embedding 2 国内高可用接入网关 v2.0

> 🎯 在国内环境下，零代码修改、全自动、批量化调用 Google Gemini Embedding 2 API

## 架构说明

```
你的应用（国内）
      ↓
FastAPI Gateway（本地 Docker，负责熔断/批处理/队列）
      ↓ 自动负载均衡 + 熔断
┌─────┬─────┬─────┐
│ SG  │ EU  │ US  │  ← nginx 透明代理（Render.com 免费部署）
└──┬──┴──┬──┴──┬──┘
   └─────┴─────┘
         ↓
generativelanguage.googleapis.com（Google API）
```

## 快速开始

见 `操作手册.md`（新手）或直接看下方命令（熟悉 Docker 的用户）

```bash
cp .env.example .env
# 编辑 .env，填写 GOOGLE_API_KEY 和 PROXY_1/2/3
docker-compose up -d
```

## 调用示例

```python
import openai
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="你在.env中设置的GATEWAY_TOKEN"
)
resp = client.embeddings.create(
    model="gemini-embedding-2-preview",
    input=["文本1", "文本2"]
)
print(len(resp.data[0].embedding))  # 3072
```

## 监控

打开浏览器访问 http://localhost:8000 查看实时节点状态。
