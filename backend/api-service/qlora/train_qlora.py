"""
QLoRA 微调脚本 - 用于简历优化领域的模型微调
使用示例：
python train_qlora.py --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
                     --data_path data/resume_knowledge/train.json \
                     --output_dir models/resume_qlora \
                     --num_train_epochs 3 \
                     --batch_size 4 \
                     --learning_rate 2e-4
"""

import os
import json
import torch
import argparse
from typing import List, Dict
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset


@dataclass
class QLoraArguments:
    """QLoRA 训练参数"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "基础模型路径"}
    )
    data_path: str = field(
        default="../../ai/train_data/train.json",
        metadata={"help": "训练数据路径"}
    )
    output_dir: str = field(
        default="models/resume_qlora",
        metadata={"help": "输出目录"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "训练轮数"}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "批次大小"}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "学习率"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度"}
    )
    lora_rank: int = field(
        default=16,
        metadata={"help": "LoRA 秩"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA alpha 参数"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "梯度累积步数"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "预热步数"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "日志步数"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "保存步数"}
    )


def load_training_data(data_path: str) -> List[Dict]:
    """加载训练数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"加载训练数据：{len(data)} 条")
    return data


def format_resume_sample(sample: Dict) -> str:
    """格式化简历样本为训练文本"""
    # 根据数据类型格式化
    if 'instruction' in sample:
        text = f"""### Instruction:
{sample['instruction']}

### Input:
{sample.get('input', '')}

### Response:
{sample['output']}
"""
    elif 'query' in sample and 'response' in sample:
        text = f"""### 用户简历:
{sample['query']}

### 优化结果:
{sample['response']}
"""
    else:
        # 通用格式
        text = json.dumps(sample, ensure_ascii=False)
    
    return text


def prepare_dataset(data: List[Dict], tokenizer: AutoTokenizer, max_length: int = 2048) -> Dataset:
    """准备训练数据集"""
    
    def tokenize_function(examples):
        texts = [format_resume_sample(sample) for sample in examples]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 训练时 labels 与 input_ids 相同（自回归语言模型）
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized
    
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def create_qlora_model(
    model_name: str,
    lora_rank: int = 16,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1
):
    """创建 QLoRA 模型"""
    
    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # 双重量化
        bnb_4bit_quant_type="nf4",  # 正态分布量化
        bnb_4bit_compute_dtype=torch.float16  # 计算精度
    )
    
    # 加载基础模型
    print(f"加载基础模型：{model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 准备模型用于 k-bit 训练
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 配置
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 注意力层
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model


def train_qlora(args: QLoraArguments):
    """执行 QLoRA 训练"""
    
    print("=" * 60)
    print("QLoRA 简历优化模型训练")
    print("=" * 60)
    
    # 1. 加载 tokenizer
    print("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载训练数据
    print("加载训练数据...")
    train_data = load_training_data(args.data_path)
    
    # 3. 准备数据集
    print("准备数据集...")
    train_dataset = prepare_dataset(train_data, tokenizer, args.max_seq_length)
    
    # 4. 创建 QLoRA 模型
    print("创建 QLoRA 模型...")
    model = create_qlora_model(
        args.model_name_or_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",  # 分页优化器，节省显存
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )
    
    # 6. 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 自回归模型，不使用 MLM
    )
    
    # 7. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    # 8. 开始训练
    print("开始训练...")
    trainer.train()
    
    # 9. 保存模型
    print("保存模型...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n训练完成！模型已保存到：{args.output_dir}")
    print(f"可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


def main():
    parser = argparse.ArgumentParser(description='QLoRA 简历优化模型训练')
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--data_path', type=str, default='../../ai/train_data/train.json')
    parser.add_argument('--output_dir', type=str, default='models/resume_qlora')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=100)
    
    args = parser.parse_args()
    
    train_qlora(args)


if __name__ == "__main__":
    main()
