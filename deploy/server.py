"""
FastAPI 推理服务
"""
import os
import json
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional
import uvicorn

app = FastAPI(title="Finance Intent API", version="1.0.0")

# 全局模型变量
model = None
tokenizer = None

# 配置
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LORA_PATH = os.getenv("LORA_PATH", "/app/models/qwen-finance-intent")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

SYSTEM_PROMPT = """你是一个金融意图识别助手。请根据用户的问题，识别意图并返回JSON格式的回复。
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


class Query(BaseModel):
    text: str
    max_new_tokens: Optional[int] = 256


class Response(BaseModel):
    intent: str
    content: str
    raw_output: str


def load_model():
    """加载模型"""
    global model, tokenizer

    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    print(f"Loading base model from {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载 LoRA 权重
    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA weights from {LORA_PATH}...")
        model = PeftModel.from_pretrained(model, LORA_PATH)

    model.eval()
    print("Model loaded successfully!")


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=Response)
async def predict(query: Query):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query.text}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=query.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 解析 JSON
    result = {
        "intent": "unknown",
        "content": response,
        "raw_output": response
    }

    try:
        response = response.strip()
        if response.startswith("{"):
            json_end = response.rfind("}") + 1
            json_str = response[:json_end]
            parsed = json.loads(json_str)
            result["intent"] = parsed.get("intent", "unknown")
            result["content"] = parsed.get("content", response)
    except json.JSONDecodeError:
        pass

    return Response(**result)


@app.post("/batch_predict")
async def batch_predict(queries: list[Query]):
    results = []
    for query in queries:
        result = await predict(query)
        results.append(result)
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
