# 三大能力建设实施计划 V2（基于现有框架补全）

## 仓库调研结论（已逐文件确认）

### 能力一：数据与知识库 pipeline —— 已有 80%，只缺最后一环

现有链路：
```
Playwright 采集 (raw_data/src/job/crawler.py，智联+51job 并发)
      ↓
job_descriptions 表 (raw_data/src/job/database.py，SQLite/MySQL 双支持)
      ↓
天池简历 CSV 导入 (raw_data/src/resume/importer.py) → resumes 表
      ↓
Flink 清洗 (data/java-pipeline：FlinkResumeJob → ResumeDataCleaner 清洗/校验 → BatchRedisSink → Redis)
      ↓
═══════════════════════════════════════════════
  ❌ 断点：Redis/MySQL 里的数据没有进入向量知识库
═══════════════════════════════════════════════
      ↓ （应补充）
bge-small-zh Embedding (rag/embeddings.py) → FAISS (rag/vector_store.py)
```

向量库现状：`scripts/manage_vector_db.py` 只能从 txt 文件入库；`generate_knowledge.py` 读的是 `ai/train_data/train.json`，**与爬虫/Flink 产出的 MySQL/Redis 数据没有打通**。

### 能力二：多模态解析 —— 现有文档解析可扩展
- `rag/document_processor.py`：仅 PDF/DOC/DOCX/TXT
- `api/routes/upload.py`：刚加了白名单（pdf/doc/docx/txt，10MB）
- 缺：图片 OCR、音频转写，以及统一入口

### 能力三：Agent 评测 —— 从零建设
- 无任何评测代码，qlora/ 只有训练
- 优势：已有 local/deepseek 双 Provider + Langfuse，天然支持双模型对比

---

## 实施方案

### 模块一：补齐知识库 pipeline 最后一环（复用现有，不重写）

**新增 1 个衔接模块 + 1 个 CLI，不动 raw_data 和 java-pipeline：**

- `rag/knowledge_builder.py`：知识库构建器
  - `build_from_mysql(source="jobs"|"resumes", ...)`：从 job_descriptions/resumes 表读取（pymysql 已有依赖）
  - `build_from_redis(...)`：读 Flink 清洗后写入 Redis 的 `resume:{id}`（redis-py 已有依赖），体现"Flink 清洗后再入向量库"的完整链路
  - 文本规范化 → 切片（按字段：JD 要求/职责、简历技能/项目）→ 复用 `embeddings.py` + `vector_store.py` 入库
  - 独立集合 `job_knowledge`（与现有 resume_knowledge、skill_router 分开）
- `scripts/build_knowledge.py`：CLI（`--source mysql|redis`、`--type jobs|resumes`、`--limit`、`--rebuild`），跑完打印入库条数 + 抽样检索验证

这样简历上"Playwright 采集 → Flink 清洗 → Embedding 切片构建向量知识库 pipeline"完整闭环，且全部复用已有代码。

### 模块二：多模态统一解析（与现有集成）

- `rag/multimodal.py`：
  - `parse_multimodal(file_path) -> {text, modality, metadata}`
  - pdf/doc/docx/txt → 复用 `document_processor.parse_file`
  - png/jpg/jpeg/bmp → RapidOCR（rapidocr-onnxruntime，ONNX CPU，~15MB）
  - mp3/wav/m4a → faster-whisper（base，CPU 离线）
  - 依赖未安装时抛友好错误（提示 pip 命令），不影响现有文档上传
- `api/routes/upload.py`：扩展名白名单加图片/音频，按类型分流到 `parse_multimodal`，返回的 text 接入现有 Agent 流程
- 可选依赖写入 `requirements-optional.txt`

### 模块三：Agent 评测框架（从零，详细设计）

**评测原理（LLM 应用评测的业界标准做法）：**

LLM 输出无法用"字符串相等"判定对错，所以用**双轨评分**：

1. **规则校验（确定性、零成本）** `evaluation/rules.py`
   - 关键词覆盖率：GT 要求的关键点（如"Python、分布式"）在输出中出现的比例
   - 结构完整性：润色结果是否保留 技能/项目/教育 等必备板块
   - 禁忌校验：是否出现空泛词（"熟练掌握"堆砌）、编造的公司/时间
   - 格式/字数区间
2. **LLM-as-Judge（语义、少量成本）** `evaluation/judge.py`
   - 用更强的模型（默认 deepseek）当"裁判"，输入 `(用户问题, 模型回答, 评分标准rubric)`
   - 三个维度各打 1-5 分并给理由：正确性、相关性、幻觉
   - 输出 JSON，解析失败重试 1 次后回退纯规则分数（不中断）

**评测集** `evaluation/datasets/eval_samples.json`：
```json
{
  "id": "score_001",
  "category": "resume_score|jd_match|polish",
  "input": "简历正文...",
  "jd": "可选",
  "ground_truth": {"must_contain": ["Python"], "structure": ["技能","项目"], "forbidden": ["精通一切"]}
}
```
准备 12-15 条覆盖三类任务的样例。语音 Ground Truth（转写测试）作为可选项预留结构，本期不实现 STT 评测。

**跑批与对比** `evaluation/runner.py` + `scripts/run_evaluation.py`：
- 同一批评测用例分别通过 `agents.registry.get_agent(provider=...)` 跑 local 和 deepseek
- 每条记录：规则通过率、Judge 三维分、耗时、token 用量（从 Langfuse/返回值取）
- 汇总指标并输出对比表：

| 指标 | 定义 |
|---|---|
| 准确率 | 规则通过率 × 0.5 + Judge 正确性归一化 × 0.5 |
| 幻觉率 | Judge 幻觉维度命中"编造"的用例占比（越低越好） |
| 相关性 | Judge 相关性平均分 |
| 平均延迟 | 每 Provider 平均响应秒数 |
| 平均 token | 每 Provider 平均消耗（成本对比） |

输出 JSON 报告到文件 + 终端 Markdown 对比表。

## 文件清单

新增：
- `rag/knowledge_builder.py`
- `rag/multimodal.py`
- `scripts/build_knowledge.py`
- `scripts/run_evaluation.py`
- `evaluation/{__init__.py,dataset.py,rules.py,judge.py,runner.py}`
- `evaluation/datasets/eval_samples.json`
- `requirements-optional.txt`

修改：
- `api/routes/upload.py`（扩展白名单 + 多模态分流）

不改动：raw_data/、data/java-pipeline/、现有向量库核心

## 测试与验证
- 纯逻辑补单测（mock LLM/OCR）：rules、dataset、knowledge_builder 的切片逻辑、multimodal 类型分流
- 采集/Flink/OCR/STT/真实评测不纳入 pytest（手动 CLI），避免重依赖
- `pytest` 全绿、前端 `tsc` 0 errors
- 手动：build_knowledge 从库入库 + 抽样检索；run_evaluation 出双模型对比表（无 deepseek key 时 local 单跑也能出结果）

## 风险
- 本地无 MySQL/Redis 数据时：knowledge_builder 支持 SQLite（raw_data 默认就是 SQLite），保证可离线演示
- OCR/STT 模型下载：可选依赖，核心零影响
- Judge 模型输出不稳定：JSON 容错 + 回退规则分
- Flink 环境（JVM/Maven）本地未必能跑：本期不要求重跑 Flink，builder 同时支持直读 MySQL 原始表和读 Redis 清洗结果两种路径
