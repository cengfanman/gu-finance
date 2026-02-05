#!/bin/bash
# AutoDL 4090 训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache
export HF_HOME=/root/autodl-tmp/cache

# 进入项目目录
cd /root/autodl-tmp/gu-finance

# 安装依赖（首次运行）
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 1. 先运行数据增强
echo "=== Step 1: Data Augmentation ==="
python scripts/data_augment.py

# 2. 开始训练
echo "=== Step 2: Training ==="
python scripts/train_lora.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --data_path ./data/train_data_augmented.json \
    --output_dir ./models/qwen-finance-intent \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_length 512 \
    --lora_r 16 \
    --lora_alpha 32

echo "=== Training Complete ==="

# 3. 测试模型
echo "=== Step 3: Testing ==="
python scripts/inference.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --lora_path ./models/qwen-finance-intent
