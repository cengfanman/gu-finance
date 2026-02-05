"""
数据增强脚本
通过同义词替换、句式变换等方式扩充训练数据
"""
import json
import random
import re
from typing import List, Dict

# 同义词替换词典
SYNONYMS = {
    "什么是": ["请问什么是", "能解释一下什么是", "帮我介绍一下", "请介绍", "什么叫"],
    "怎么": ["如何", "怎样", "该怎么", "应该怎么"],
    "有什么": ["有哪些", "都有什么", "包括哪些"],
    "区别": ["差别", "不同", "差异", "区分"],
    "好吗": ["怎么样", "如何", "可以吗", "靠谱吗"],
    "可以": ["能", "能不能", "可不可以", "行不行"],
    "推荐": ["建议", "介绍", "有什么好的"],
    "查一下": ["帮我查", "查询", "看看", "帮我看看"],
    "现在": ["目前", "当前", "眼下"],
    "股票": ["个股", "股"],
    "基金": ["基"],
    "收益": ["回报", "收益率", "赚钱"],
    "风险": ["危险", "不安全", "可能亏"],
    "涨": ["上涨", "涨起来", "往上走"],
    "跌": ["下跌", "跌下去", "往下走"],
    "买": ["买入", "购买", "入手"],
    "卖": ["卖出", "出手", "抛"],
    "投资": ["理财", "配置"],
    "开户": ["开个户", "办理开户", "注册账户"],
    "条件": ["要求", "门槛", "需要什么"],
    "怎么样": ["如何", "好不好", "咋样"],
    "多少": ["几多", "是多少", "有多少"],
}

# 口语化变换
COLLOQUIAL_PATTERNS = [
    (r"^什么是(.+)[？?]?$", ["{0}是啥？", "{0}是什么意思？", "啥是{0}？", "{0}啥意思？"]),
    (r"^(.+)怎么样[？?]?$", ["{0}咋样？", "{0}好不好？", "{0}行不行？"]),
    (r"^如何(.+)[？?]?$", ["怎么{0}？", "咋{0}？", "{0}要怎么做？"]),
    (r"^(.+)有什么(.+)[？?]?$", ["{0}有啥{1}？", "{0}都有哪些{1}？"]),
]

# 前缀变换
PREFIXES = [
    "", "请问", "想问一下", "帮我看看", "问一下", "请教一下",
    "我想知道", "能告诉我", "麻烦问下", "有个问题"
]

# 后缀变换
SUFFIXES = [
    "", "？", "呢？", "啊？", "吗？", "呀？"
]

# 股票代码和名称映射
STOCK_NAMES = {
    "NVDA": ["英伟达", "老黄家的", "黄仁勋"],
    "AAPL": ["苹果", "苹果公司"],
    "TSLA": ["特斯拉", "马斯克家的"],
    "GOOGL": ["谷歌", "Google"],
    "MSFT": ["微软"],
    "AMZN": ["亚马逊"],
    "META": ["Meta", "脸书"],
    "茅台": ["贵州茅台", "600519"],
    "腾讯": ["腾讯控股", "00700"],
    "阿里": ["阿里巴巴", "BABA"],
    "比亚迪": ["002594", "BYD"],
    "宁德时代": ["300750", "宁德"],
    "中芯国际": ["688981", "中芯"],
}

# 城市列表（用于天气类问题）
CITIES = [
    "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京",
    "西安", "重庆", "天津", "苏州", "郑州", "长沙", "青岛", "厦门"
]

def synonym_replace(text: str) -> List[str]:
    """同义词替换"""
    results = [text]
    for word, synonyms in SYNONYMS.items():
        if word in text:
            for syn in synonyms:
                new_text = text.replace(word, syn, 1)
                if new_text != text:
                    results.append(new_text)
    return list(set(results))

def colloquial_transform(text: str) -> List[str]:
    """口语化变换"""
    results = []
    for pattern, templates in COLLOQUIAL_PATTERNS:
        match = re.match(pattern, text)
        if match:
            groups = match.groups()
            for template in templates:
                try:
                    new_text = template.format(*groups)
                    results.append(new_text)
                except:
                    pass
    return results

def add_prefix_suffix(text: str) -> List[str]:
    """添加前缀后缀"""
    results = []
    # 去除原有标点
    clean_text = text.rstrip("？?。.！!")

    for prefix in random.sample(PREFIXES, min(3, len(PREFIXES))):
        for suffix in random.sample(SUFFIXES, min(2, len(SUFFIXES))):
            if prefix:
                new_text = f"{prefix}，{clean_text}{suffix}"
            else:
                new_text = f"{clean_text}{suffix}"
            results.append(new_text)
    return results

def stock_name_replace(text: str) -> List[str]:
    """股票名称替换"""
    results = []
    for code, names in STOCK_NAMES.items():
        if code in text:
            for name in names:
                new_text = text.replace(code, name)
                results.append(new_text)
        for name in names:
            if name in text:
                new_text = text.replace(name, code)
                results.append(new_text)
                for other_name in names:
                    if other_name != name:
                        new_text = text.replace(name, other_name)
                        results.append(new_text)
    return list(set(results))

def city_replace(text: str) -> List[str]:
    """城市名替换（用于天气类）"""
    results = []
    for city in CITIES:
        if city in text:
            for new_city in random.sample(CITIES, min(5, len(CITIES))):
                if new_city != city:
                    new_text = text.replace(city, new_city)
                    results.append(new_text)
            break
    return results

def augment_sample(sample: Dict) -> List[Dict]:
    """对单条数据进行增强"""
    instruction = sample["instruction"]
    output = sample["output"]
    intent = json.loads(output).get("intent", "")

    augmented = [sample]  # 保留原始数据

    # 同义词替换
    syn_results = synonym_replace(instruction)
    for new_inst in syn_results[:3]:  # 限制数量
        if new_inst != instruction:
            augmented.append({
                "instruction": new_inst,
                "input": "",
                "output": output
            })

    # 口语化变换
    colloquial_results = colloquial_transform(instruction)
    for new_inst in colloquial_results[:2]:
        augmented.append({
            "instruction": new_inst,
            "input": "",
            "output": output
        })

    # 股票名称替换（仅对股票相关intent）
    if intent in ["stock_prediction", "stock_query"]:
        stock_results = stock_name_replace(instruction)
        for new_inst in stock_results[:3]:
            augmented.append({
                "instruction": new_inst,
                "input": "",
                "output": output
            })

    # 城市替换（仅对天气intent）
    if intent == "weather":
        city_results = city_replace(instruction)
        for new_inst in city_results[:3]:
            augmented.append({
                "instruction": new_inst,
                "input": "",
                "output": output
            })

    # 前缀后缀变换（随机选择部分样本）
    if random.random() < 0.3:
        prefix_results = add_prefix_suffix(instruction)
        for new_inst in random.sample(prefix_results, min(2, len(prefix_results))):
            augmented.append({
                "instruction": new_inst,
                "input": "",
                "output": output
            })

    return augmented

def augment_dataset(input_file: str, output_file: str, target_size: int = 500):
    """增强整个数据集"""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"原始数据量: {len(data)}")

    augmented_data = []
    for sample in data:
        augmented = augment_sample(sample)
        augmented_data.extend(augmented)

    # 去重
    seen = set()
    unique_data = []
    for item in augmented_data:
        key = item["instruction"]
        if key not in seen:
            seen.add(key)
            unique_data.append(item)

    print(f"增强后数据量: {len(unique_data)}")

    # 如果数据量不足，继续增强
    if len(unique_data) < target_size:
        print(f"数据量不足 {target_size}，继续增强...")
        additional = []
        for sample in unique_data:
            prefix_results = add_prefix_suffix(sample["instruction"])
            for new_inst in prefix_results[:2]:
                if new_inst not in seen:
                    seen.add(new_inst)
                    additional.append({
                        "instruction": new_inst,
                        "input": "",
                        "output": sample["output"]
                    })
                if len(unique_data) + len(additional) >= target_size:
                    break
            if len(unique_data) + len(additional) >= target_size:
                break
        unique_data.extend(additional)

    # 打乱顺序
    random.shuffle(unique_data)

    # 保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)

    print(f"最终数据量: {len(unique_data)}")
    print(f"已保存至: {output_file}")

    # 统计各意图数量
    intent_count = {}
    for item in unique_data:
        try:
            intent = json.loads(item["output"]).get("intent", "unknown")
            intent_count[intent] = intent_count.get(intent, 0) + 1
        except:
            pass

    print("\n各意图数据分布:")
    for intent, count in sorted(intent_count.items(), key=lambda x: -x[1]):
        print(f"  {intent}: {count}")

if __name__ == "__main__":
    import os

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    input_file = os.path.join(project_dir, "data", "train_data.json")
    output_file = os.path.join(project_dir, "data", "train_data_augmented.json")

    augment_dataset(input_file, output_file, target_size=500)
