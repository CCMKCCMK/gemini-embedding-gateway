# ============================================================
# Gemini Proxy Node - Dockerfile
# 基于 nginx:alpine，镜像仅 ~8MB
# 部署到 Render / Railway / Koyeb 等平台
# ============================================================
FROM nginx:1.27-alpine

# 安装 wget（用于健康检查）
RUN apk add --no-cache wget

# 拷贝配置
COPY nginx.conf /etc/nginx/nginx.conf

# Render 等平台要求监听 10000 端口
EXPOSE 10000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:10000/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
