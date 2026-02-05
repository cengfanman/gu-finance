"""
推理脚本 - 加载微调后的模型进行意图识别
"""
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

class FinanceIntentClassifier:
    """金融意图识别器"""

    def __init__(self, base_model_path: str, lora_path: str = None, device: str = "auto"):
        """
        初始化模型

        Args:
            base_model_path: 基础模型路径
            lora_path: LoRA权重路径，如果为None则使用base_model_path
            device: 设备，"auto"/"cuda"/"cpu"
        """
        print(f"Loading tokenizer from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )

        print(f"Loading base model from {base_model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

        # 加载LoRA权重
        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA weights from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.eval()

        # System prompt
        self.system_prompt = """你是一个金融意图识别助手。请根据用户的问题，识别意图并返回JSON格式的回复。
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

    def predict(self, query: str, max_new_tokens: int = 256) -> dict:
        """
        预测意图

        Args:
            query: 用户问题
            max_new_tokens: 最大生成token数

        Returns:
            {"intent": str, "content": str, "raw_output": str}
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]

        # 构建输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 使用贪婪解码保证结果一致性
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 解码
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 解析JSON
        result = {
            "intent": "unknown",
            "content": response,
            "raw_output": response
        }

        try:
            # 尝试提取JSON
            response = response.strip()
            if response.startswith("{"):
                json_end = response.rfind("}") + 1
                json_str = response[:json_end]
                parsed = json.loads(json_str)
                result["intent"] = parsed.get("intent", "unknown")
                result["content"] = parsed.get("content", response)
        except json.JSONDecodeError:
            pass

        return result

    def predict_batch(self, queries: list, max_new_tokens: int = 256) -> list:
        """
        批量预测

        Args:
            queries: 问题列表
            max_new_tokens: 最大生成token数

        Returns:
            预测结果列表
        """
        results = []
        for query in queries:
            result = self.predict(query, max_new_tokens)
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description="Finance Intent Classification Inference")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model path")
    parser.add_argument("--lora_path", type=str, default="./models/qwen-finance-intent",
                        help="LoRA weights path")
    parser.add_argument("--query", type=str, default=None,
                        help="Single query to test")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")
    args = parser.parse_args()

    # 初始化模型
    classifier = FinanceIntentClassifier(
        base_model_path=args.base_model,
        lora_path=args.lora_path
    )

    # 测试用例
    test_queries = [
        "什么是ETF？",
        "茅台股票明天会涨吗？",
        "今天北京天气怎么样？",
        "怎么开通科创板？",
        "推荐一部电影",
        "基金定投好还是一次性买入好？",
        "美联储加息对A股有什么影响？",
        "新手买什么基金比较好？",
        "杠杆交易风险大吗？",
        "腾讯现在股价多少？"
    ]

    if args.query:
        # 单条测试
        result = classifier.predict(args.query)
        print(f"\n问题: {args.query}")
        print(f"意图: {result['intent']}")
        print(f"回复: {result['content']}")

    elif args.interactive:
        # 交互模式
        print("\n=== 金融意图识别交互模式 ===")
        print("输入问题进行测试，输入 'quit' 退出\n")

        while True:
            query = input("请输入问题: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break
            if not query:
                continue

            result = classifier.predict(query)
            print(f"意图: {result['intent']}")
            print(f"回复: {result['content']}")
            print()

    else:
        # 批量测试
        print("\n=== 测试用例结果 ===\n")
        for query in test_queries:
            result = classifier.predict(query)
            print(f"问题: {query}")
            print(f"意图: {result['intent']}")
            print(f"回复: {result['content']}")
            print("-" * 50)


if __name__ == "__main__":
    main()
