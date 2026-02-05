#!/bin/bash
# 构建和运行 Docker 容器

set -e

echo "=== 构建 Docker 镜像 ==="
cd "$(dirname "$0")/.."

# 检查模型是否存在
if [ ! -d "models/qwen-finance-intent" ]; then
    echo "错误: 模型目录不存在，请先完成训练"
    echo "模型应该在: models/qwen-finance-intent/"
    exit 1
fi

# 构建镜像
docker build -f deploy/Dockerfile -t finance-intent-api:latest .

echo "=== 运行容器 ==="
docker run -d \
    --name finance-intent-api \
    --gpus all \
    -p 8000:8000 \
    -e HF_ENDPOINT=https://hf-mirror.com \
    -v $(pwd)/models:/app/models \
    finance-intent-api:latest

echo "=== 服务已启动 ==="
echo "API 地址: http://localhost:8000"
echo "健康检查: http://localhost:8000/health"
echo "API 文档: http://localhost:8000/docs"

echo ""
echo "测试命令:"
echo 'curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '\''{"text": "什么是ETF？"}'\'''
