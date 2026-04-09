"""
准备简历优化领域的 QLoRA 微调数据
"""

import json
import os
import re
from pathlib import Path


def load_resume_knowledge() -> list:
    """从知识库文件加载简历优化样本"""
    
    knowledge_files = [
        "data/resume_knowledge/training_samples_knowledge.txt",
        "data/resume_knowledge/additional_samples.txt"
    ]
    
    samples = []
    
    for file_path in knowledge_files:
        if not os.path.exists(file_path):
            print(f"文件不存在：{file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析样本（根据实际格式调整）
        # 假设格式：### 示例 X：XXX\n**原始:** ...\n**优化:**\n技能：...\n项目简介：...
        
        # 提取示例
        pattern = r'### (示例 \d+：.*?)\n\*\*原始:\*\* (.*?)\n\*\*优化:\*\*\n(.*?)(?=\n### |\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            example_title, original, optimized = match
            
            # 构建训练样本
            sample = {
                "instruction": "请优化这份简历的技能栏和项目描述",
                "input": original.strip(),
                "output": optimized.strip(),
                "category": extract_category(example_title)
            }
            samples.append(sample)
    
    return samples


def extract_category(title: str) -> str:
    """从标题提取类别"""
    categories = {
        "Python": "后端开发",
        "Java": "后端开发",
        "Go": "后端开发",
        "C++": "后端开发",
        "Node.js": "后端开发",
        "Vue": "前端开发",
        "React": "前端开发",
        "移动端": "前端开发",
        "机器学习": "算法",
        "深度学习": "算法",
        "计算机视觉": "算法",
        "NLP": "算法",
        "大模型": "算法",
        "AIGC": "算法",
        "数据": "数据方向",
        "测试": "测试开发",
        "运维": "运维开发",
        "SRE": "运维开发",
        "安全": "网络安全",
        "管理": "技术管理"
    }
    
    for keyword, category in categories.items():
        if keyword in title:
            return category
    
    return "其他"


def create_chat_format_sample(sample: dict) -> dict:
    """转换为对话格式的训练样本"""
    
    return {
        "messages": [
            {"role": "system", "content": "你是一位专业的简历优化专家，擅长根据用户现有技能提供精准的优化建议。"},
            {"role": "user", "content": f"请优化我的简历：\n{sample['input']}"},
            {"role": "assistant", "content": sample['output']}
        ],
        "category": sample['category']
    }


def prepare_training_data(output_path: str = "../../ai/train_data/train.json"):
    """准备训练数据"""
    
    print("加载简历知识库...")
    samples = load_resume_knowledge()
    
    print(f"共加载 {len(samples)} 个样本")
    
    # 统计类别分布
    category_count = {}
    for sample in samples:
        cat = sample['category']
        category_count[cat] = category_count.get(cat, 0) + 1
    
    print("\n类别分布:")
    for cat, count in sorted(category_count.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    # 转换为对话格式
    chat_samples = [create_chat_format_sample(s) for s in samples]
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chat_samples, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练数据已保存到：{output_path}")
    print(f"总计：{len(chat_samples)} 条")
    
    # 显示示例
    print("\n示例数据:")
    for i, sample in enumerate(chat_samples[:2]):
        print(f"\n--- 示例 {i+1} ---")
        print(f"类别：{sample['category']}")
        print(f"用户：{sample['messages'][1]['content'][:100]}...")
        print(f"助手：{sample['messages'][2]['content'][:100]}...")


if __name__ == "__main__":
    prepare_training_data()
