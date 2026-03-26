import torch
import torch.nn as nn
import argparse
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model


print("=== 知识蒸馏脚本（完全本地无联网）===")
teacher_path = input("请输入老师模型路径: ").strip()
student_path = input("请输入学生模型保存路径: ").strip()
doc_path = input("请输入本地训练文档路径 (.txt): ").strip()
student_base_path = input("请输入【学生基础模型】本地路径: ").strip()

print(f"\n📋 配置确认:")
print(f"  老师模型: {teacher_path}")
print(f"  学生基础模型: {student_base_path}")
print(f"  最终学生保存: {student_path}")
print(f"  训练文档: {doc_path}")
print("\n开始初始化...\n")


# 4bit 量化（修复你之前的报错）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


# 加载 Teacher（本地）
tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)

teacher = AutoModelForCausalLM.from_pretrained(
    teacher_path,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)
teacher.eval()
teacher.requires_grad_(False)


# 加载 Student
student = AutoModelForCausalLM.from_pretrained(
    student_base_path,
    device_map="auto",
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"
)
student = get_peft_model(student, lora_config)


# 蒸馏损失
class DistillLoss(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        student_out = self.student(**inputs)
        s_logits = student_out.logits

        with torch.no_grad():
            teacher_out = self.teacher(**inputs)
            t_logits = teacher_out.logits

        loss = self.ce_loss(
            s_logits.reshape(-1, s_logits.size(-1)),
            t_logits.argmax(dim=-1).reshape(-1)
        )
        return loss

distill_loss = DistillLoss(teacher, student)


# 自定义 Trainer
class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = distill_loss(inputs)
        return (loss, None) if return_outputs else loss


# 读取本地文档
with open(doc_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

train_data = [{"text": line} for line in lines]
dataset = Dataset.from_list(train_data)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

token_dataset = dataset.map(tokenize_fn)


# 训练参数
training_args = TrainingArguments(
    output_dir="./distill_first_run",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=5,
    fp16=True,
    optim="paged_adamw_8bit",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=5,
    report_to="none"
)


# 启动蒸馏
trainer = DistillTrainer(
    model=student,
    args=training_args,
    train_dataset=token_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()

student.save_pretrained(student_path)
tokenizer.save_pretrained(student_path)

print(f"\n✅ 首次蒸馏完成！")
print(f"📁 模型已保存到：{student_path}")