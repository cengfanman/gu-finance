"""
Qwen 7B LoRA 微调训练脚本
适用于 AutoDL 4090 24G 显存
"""
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Qwen LoRA Fine-tuning")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--data_path", type=str, default="./data/train_data_augmented.json",
                        help="Training data path")
    parser.add_argument("--output_dir", type=str, default="./models/qwen-finance-intent",
                        help="Output directory for model")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    """加载模型和tokenizer"""
    print(f"Loading model: {args.model_name}")

    # 量化配置
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization")
    elif args.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # 如果使用量化，准备模型
    if args.use_4bit or args.use_8bit:
        model = prepare_model_for_kbit_training(model)

    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

def preprocess_function(example, tokenizer, max_length):
    """数据预处理"""
    instruction = example["instruction"]
    output = example["output"]

    # 构建对话格式
    system_prompt = """你是一个金融意图识别助手。请根据用户的问题，识别意图并返回JSON格式的回复。
意图类别包括：
- finance_knowledge: 金融基础知识问答
- stock_prediction: 股票预测类（需免责声明）
- stock_query: 股票数据查询
- account_service: 账户服务咨询
- investment_guide: 投资指导建议
- risk_warning: 风险提示
- macro_economy: 宏观经济
- weather: 天气问题（拒绝回答）
- other: 其他非金融问题（拒绝回答）

回复格式：{"intent": "意图类别", "content": "回复内容"}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output}
    ]

    # 使用tokenizer的chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    # 设置labels（用于计算loss）
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

def main():
    args = get_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # 加载数据
    print(f"Loading data from: {args.data_path}")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    print(f"Dataset size: {len(dataset)}")

    # 数据预处理
    print("Preprocessing data...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    # 划分训练集和验证集
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="adamw_torch",
        report_to="none",  # 如果要用wandb，改成"wandb"
        save_total_limit=3,
        dataloader_num_workers=4,
        gradient_checkpointing=True,  # 节省显存
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    print("Starting training...")
    trainer.train()

    # 保存模型
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # 保存训练配置
    config = {
        "base_model": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Training completed!")

if __name__ == "__main__":
    main()
