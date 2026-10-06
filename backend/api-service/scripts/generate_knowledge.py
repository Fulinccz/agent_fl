import json
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def generate_knowledge_base():
    input_file = os.path.join(os.path.dirname(__file__), '..', '..', 'ai', 'train_data', 'train.json')
    output_file = os.path.join(os.path.dirname(__file__), 'data', 'resume_knowledge', 'training_samples_knowledge.txt')
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    content = []
    content.append("# 简历优化训练样本知识库\n")
    content.append(f"本文件基于 {len(data)} 条真实训练样本生成，包含简历优化、JD匹配等最佳实践。\n")
    
    # 分类整理
    resume_samples = []  # 简历优化样本
    jd_match_samples = []  # JD匹配样本
    
    for item in data:
        inp = item.get('input', '')
        out = item.get('output', '')
        
        if inp.startswith('JD：'):
            jd_match_samples.append({'input': inp, 'output': out})
        else:
            resume_samples.append({'input': inp, 'output': out})
    
    # 第一部分：简历优化最佳实践
    content.append("=" * 80)
    content.append("# 第一部分：简历优化最佳实践")
    content.append("=" * 80 + "\n")
    
    content.append("## 1.1 后端开发方向\n")
    for sample in resume_samples:
        inp = sample['input']
        out = sample['output']
        if any(kw in inp for kw in ['后端', 'Go', 'Java', 'Python', 'API', '接口']):
            content.append(f"### 原始简历\n{inp}\n")
            content.append(f"### 优化结果\n{out}\n")
            content.append("-" * 60 + "\n")
    
    content.append("\n## 1.2 数据开发方向\n")
    for sample in resume_samples:
        inp = sample['input']
        out = sample['output']
        if any(kw in inp for kw in ['数据', 'Hive', 'Spark', 'Flink', '数仓', 'ETL']):
            content.append(f"### 原始简历\n{inp}\n")
            content.append(f"### 优化结果\n{out}\n")
            content.append("-" * 60 + "\n")
    
    content.append("\n## 1.3 测试/运维方向\n")
    for sample in resume_samples:
        inp = sample['input']
        out = sample['output']
        if any(kw in inp for kw in ['测试', '运维', 'Jenkins', '自动化', 'ELK', 'Linux']):
            content.append(f"### 原始简历\n{inp}\n")
            content.append(f"### 优化结果\n{out}\n")
            content.append("-" * 60 + "\n")
    
    content.append("\n## 1.4 AI/算法方向\n")
    for sample in resume_samples:
        inp = sample['input']
        out = sample['output']
        if any(kw in inp for kw in ['LLM', 'RAG', 'LangChain', '模型', 'AI', '推荐', '广告算法']):
            content.append(f"### 原始简历\n{inp}\n")
            content.append(f"### 优化结果\n{out}\n")
            content.append("-" * 60 + "\n")
    
    content.append("\n## 1.5 前端/其他方向\n")
    for sample in resume_samples:
        inp = sample['input']
        out = sample['output']
        if not any(kw in inp for kw in ['后端', '数据', '测试', '运维', 'LLM', 'RAG', 'LangChain']):
            content.append(f"### 原始简历\n{inp}\n")
            content.append(f"### 优化结果\n{out}\n")
            content.append("-" * 60 + "\n")
    
    # 第二部分：JD匹配与简历改写
    content.append("\n" + "=" * 80)
    content.append("# 第二部分：JD匹配与简历改写案例")
    content.append("=" * 80 + "\n")
    
    content.append("## 2.1 JD匹配核心原则\n")
    content.append("""
### 匹配原则：
1. **关键词对齐**：确保简历包含JD中的核心技术栈关键词
2. **能力量化**：将JD要求的技能转化为具体的项目经验描述
3. **深度展示**：不仅提到使用过某技术，还要体现掌握程度和应用场景
4. **成果导向**：用具体数字和成果证明能力符合JD要求

### 改写技巧：
- 将"了解XX"改为"熟练使用XX完成YY任务"
- 将"参与过XX项目"改为"负责XX模块开发，实现ZZ效果"
- 补充JD要求但简历缺失的关键技术点
- 使用行业通用术语和专业表达
""")
    
    for i, sample in enumerate(jd_match_samples, 1):
        inp = sample['input']
        out = sample['output']
        content.append(f"\n### 案例{i}: JD匹配改写\n")
        content.append(f"**JD要求 + 原始简历**\n{inp}\n")
        content.append(f"**匹配改写结果**\n{out}\n")
        content.append("-" * 60 + "\n")
    
    # 第三部分：优化规则总结
    content.append("\n" + "=" * 80)
    content.append("# 第三部分：简历优化核心规则总结")
    content.append("=" * 80 + "\n")
    
    content.append("""
## 3.1 技术栈提取规范
- 从原始简历中准确识别所有技术关键词（编程语言、框架、工具）
- 按类别分组：语言、框架、数据库、中间件、工具链
- 确保与目标岗位JD的技术栈高度匹配

## 3.2 技能描述规范
- 使用动作动词开头（负责、主导、开发、设计、实现、优化、提升）
- 每个技能点配合具体应用场景
- 区分"熟练使用"、"精通"、"了解"等掌握程度

## 3.3 项目简介撰写规范
- 采用STAR法则（情境-任务-行动-结果）
- 必须包含量化指标（性能提升%、用户量、响应时间等）
- 突出个人贡献和技术难点
- 字数控制在100-200字之间

## 3.4 常见问题修正
| 问题类型 | 错误示例 | 正确示例 |
|---------|---------|---------|
| 描述笼统 | "负责后端开发" | "负责核心交易系统后端开发，日均处理订单50万+" |
| 缺少量化 | "提升了性能" | "将接口响应时间从500ms降低到100ms，提升80%" |
| 技术不明确 | "做过优化" | "通过Redis缓存优化降低DB压力70%" |
| 职责不清 | "参与项目" | "主导XX模块设计，独立完成从0到1的开发" |

## 3.5 不同岗位优化重点
- **后端开发**：突出高并发、微服务、数据库优化、系统稳定性
- **数据开发**：强调数据处理量、数据质量、效率提升、数仓架构
- **测试开发**：展示自动化覆盖率、bug发现数量、效率提升比例
- **AI/算法**：体现模型效果指标、落地场景、技术创新点
- **前端开发**：注重性能优化、用户体验、组件化能力

---
*本文档由训练样本自动生成，用于RAG检索增强，为简历优化提供参考知识。*
""")
    
    # 写入文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    
    print(f"✅ 知识库文档生成成功！")
    print(f"   文件路径: {output_file}")
    print(f"   总样本数: {len(data)} 条")
    print(f"   - 简历优化: {len(resume_samples)} 条")
    print(f"   - JD匹配: {len(jd_match_samples)} 条")

if __name__ == '__main__':
    generate_knowledge_base()