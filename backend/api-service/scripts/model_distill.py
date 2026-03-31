import torch
import warnings
import torch.nn as nn
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# ======================
# 关闭所有警告
# ======================
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# ======================
# 交互输入路径
# ======================
print("=== 知识蒸馏脚本 ===")
teacher_path = input("请输入老师模型路径: ").strip()
student_path = input("请输入学生模型保存路径: ").strip()
doc_path = input("请输入本地文档路径 (.txt): ").strip()
student_base_path = input("请输入学生基础模型路径: ").strip()

print(f"\n📋 配置确认:")
print(f"  老师模型: {teacher_path}")
print(f"  学生基础模型: {student_base_path}")
print(f"  学生模型保存: {student_path}")
print(f"  训练文档: {doc_path}")
print("\n开始初始化...\n")

# ======================
# 加载 Teacher
# ======================
tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)

teacher = AutoModelForCausalLM.from_pretrained(
    teacher_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float32
)
teacher.eval()
teacher.requires_grad_(False)

# ======================
# 加载 Student
# ======================
student = AutoModelForCausalLM.from_pretrained(
    student_base_path,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float32
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"
)
student = get_peft_model(student, lora_config)

# ======================
# 蒸馏损失
# ======================
class DistillLoss(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def forward(self, inputs):
        import torch.nn.functional as F
        student_out = self.student(**inputs)
        s_logits = student_out.logits

        with torch.no_grad():
            teacher_out = self.teacher(**inputs)
            t_logits = teacher_out.logits

        # Soft label 蒸馏
        T = 2.0  # 蒸馏温度，可根据实际调整
        s_log_probs = F.log_softmax(s_logits / T, dim=-1)
        t_probs = F.softmax(t_logits / T, dim=-1)
        loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (T * T)
        return loss

distill_loss = DistillLoss(teacher, student)

# ======================
# 修复 compute_loss
# ======================
class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = distill_loss(inputs)
        return (loss, None) if return_outputs else loss

# ======================
# 加载本地文档
# ======================
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

# ======================
# 训练参数（无报错版）
# ======================
training_args = TrainingArguments(
    output_dir="./distill_first_run",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=5,
    fp16=False,
    use_cpu=True,
    optim="adamw_torch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=5,
    report_to="none"
)

# ======================
# 启动蒸馏
# ======================
trainer = DistillTrainer(
    model=student,
    args=training_args,
    train_dataset=token_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()

# ======================
# 保存模型
# ======================
student.save_pretrained(student_path)
tokenizer.save_pretrained(student_path)

print(f"\n✅ 首次蒸馏完成！")
print(f"📁 模型已保存到：{student_path}")