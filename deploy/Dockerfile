# Qwen Finance Intent 推理服务
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 安装依赖
RUN pip install --no-cache-dir \
    transformers>=4.37.0 \
    peft>=0.7.0 \
    accelerate>=0.25.0 \
    bitsandbytes>=0.41.0 \
    fastapi \
    uvicorn \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制模型和代码
COPY models/qwen-finance-intent /app/models/qwen-finance-intent
COPY deploy/server.py /app/server.py

# 设置环境变量
ENV HF_ENDPOINT=https://hf-mirror.com
ENV TRANSFORMERS_CACHE=/app/cache

EXPOSE 8000

CMD ["python", "server.py"]
