"""
简历数据增强脚本
根据技能、熟练度、工作年限生成具体的工作描述和项目描述
直接覆盖原文件
"""
import csv
import random
import re
from pathlib import Path
from io import StringIO

# 项目描述模板库
PROJECT_TEMPLATES = {
    'Python': [
        "使用Python开发数据处理Pipeline，日均处理数据量达{num}万条，通过优化算法将处理效率提升{pct}%",
        "基于Python+Django搭建后台管理系统，支撑{num}万用户并发访问，接口平均响应时间{pct}ms",
        "使用Python编写自动化运维脚本，将部署效率提升{pct}%，故障恢复时间缩短至{num}分钟",
    ],
    'Java': [
        "负责Java后端服务开发，基于Spring Cloud搭建微服务架构，支撑日活用户{num}万",
        "主导Java系统性能优化，通过JVM调优和缓存策略，接口响应时间降低{pct}%",
        "使用Java开发分布式任务调度平台，日均处理任务量达{num}万，成功率99.9%",
    ],
    'Go': [
        "使用Go开发高并发网关服务，单机QPS达{num}万，内存占用降低{pct}%",
        "基于Go实现分布式消息推送系统，日推送量达{num}万条，延迟控制在{pct}ms内",
    ],
    'JavaScript': [
        "使用JavaScript+Vue开发企业级后台管理系统，组件复用率达{pct}%，开发效率提升{num}%",
        "基于React构建高性能前端页面，首屏加载时间优化至{pct}秒内，用户留存率提升{num}%",
    ],
    'SQL': [
        "负责数据库架构设计，优化慢查询{pct}条，SQL执行效率提升{num}倍",
        "设计分库分表方案，支撑数据量达{num}亿级，查询性能提升{pct}%",
    ],
    'Redis': [
        "设计Redis缓存架构，缓存命中率提升至{pct}%，DB负载降低{num}%",
        "基于Redis实现分布式锁和限流，支撑秒杀活动峰值QPS达{num}万",
    ],
    'Kafka': [
        "搭建Kafka消息队列集群，支撑日均{num}万条消息吞吐，消费延迟控制在{pct}ms内",
        "基于Kafka实现日志采集系统，日采集量达{num}TB，数据丢失率为0",
    ],
    'Flink': [
        "使用Flink构建实时数仓，日均处理数据量达{num}亿条，延迟控制在{pct}秒内",
        "基于Flink实现实时风控系统，规则命中率达{pct}%，拦截风险交易{num}万笔",
    ],
    'TensorFlow/PyTorch': [
        "使用TensorFlow搭建推荐模型，CTR提升{pct}%，日均带来GMV增长{num}万",
        "基于PyTorch实现图像识别系统，准确率达{pct}%，日处理图片{num}万张",
    ],
    'Hadoop/Spark': [
        "搭建Hadoop+Spark大数据平台，支撑PB级数据存储，日均处理数据量达{num}TB",
        "使用Spark优化ETL流程，任务执行时间缩短{pct}%，资源利用率提升{num}%",
    ],
    'Docker/Kubernetes': [
        "搭建Kubernetes容器云平台，管理{num}个微服务，部署效率提升{pct}%",
        "基于Docker实现CI/CD流水线，构建时间缩短至{pct}分钟，发布频率提升{num}倍",
    ],
    'Spring Boot': [
        "使用Spring Boot开发核心业务系统，接口数达{num}个，日均调用量{pct}万次",
        "基于Spring Boot搭建中台服务，支撑{num}个业务线，代码复用率提升{pct}%",
    ],
    'MySQL': [
        "负责MySQL集群运维，优化慢查询{pct}条，主从延迟控制在{num}ms内",
        "设计MySQL高可用架构，实现自动故障切换，RTO小于{num}分钟",
    ],
    'MongoDB': [
        "使用MongoDB存储非结构化数据，支撑{num}亿文档，查询性能提升{pct}%",
    ],
    'Elasticsearch': [
        "搭建Elasticsearch搜索集群，支撑日均{num}万次搜索请求，响应时间{pct}ms",
    ],
    'Linux': [
        "负责Linux服务器运维，管理{num}台服务器，系统可用性达{pct}%",
    ],
    'AWS/Azure': [
        "基于AWS云原生架构迁移，成本降低{pct}%，系统可用性提升至{num}个9",
    ],
    'React/Vue': [
        "使用React开发大型单页应用，组件化率达{pct}%，代码复用提升{num}%",
    ],
    'HTML/CSS': [
        "负责前端页面重构，性能评分提升至{pct}分，首屏加载时间降低{num}%",
    ],
    'Pandas/Numpy': [
        "使用Pandas处理数据分析任务，日均处理{num}万条记录，分析效率提升{pct}%",
    ],
    'Tableau/Power BI/FineBI': [
        "搭建数据可视化平台，支撑{num}个业务看板，数据决策效率提升{pct}%",
    ],
    'Selenium': [
        "使用Selenium搭建自动化测试框架，覆盖{pct}%核心场景，测试效率提升{num}倍",
    ],
    'JMeter': [
        "使用JMeter进行性能压测，发现瓶颈{pct}处，系统容量提升{num}倍",
    ],
    'Postman': [
        "使用Postman管理API接口{num}个，接口文档覆盖率{pct}%",
    ],
    'JUnit': [
        "使用JUnit编写单元测试，代码覆盖率达{pct}%，Bug率降低{num}%",
    ],
    'RESTful API': [
        "设计RESTful API规范，定义接口{num}个，调用成功率{pct}%",
    ],
    'Node.js': [
        "使用Node.js开发BFF层，支撑{num}万并发连接，响应时间{pct}ms",
    ],
    'TypeScript': [
        "使用TypeScript重构前端项目，类型覆盖率达{pct}%，Bug率降低{num}%",
    ],
    'Webpack': [
        "优化Webpack构建配置，构建时间缩短{pct}%，产物体积减少{num}%",
    ],
    'Ansible': [
        "使用Ansible实现自动化运维，管理{num}台服务器，部署效率提升{pct}%",
    ],
    'Terraform': [
        "使用Terraform管理云基础设施，资源创建效率提升{pct}%，配置错误降低{num}%",
    ],
    'Django/Flask': [
        "使用Django开发后台管理系统，支撑{num}万用户，接口响应时间{pct}ms",
    ],
    'RabbitMQ': [
        "搭建RabbitMQ消息队列，支撑日均{num}万条消息，投递成功率{pct}%",
    ],
}

# 工作总览模板
WORK_OVERVIEW_TEMPLATES = [
    "{years}年互联网开发经验，精通{skills}，具备丰富的大型项目架构设计经验",
    "{years}年技术积累，擅长{skills}，主导过多个高并发、高可用系统的设计与实现",
    "拥有{years}年开发经验，技术栈涵盖{skills}，在分布式系统和微服务架构方面有深入实践",
    "{years}年资深开发工程师，精通{skills}，具备从0到1搭建技术体系的能力",
]

# 项目职责模板
PROJECT_ROLE_TEMPLATES = [
    "作为核心技术成员，负责系统架构设计和技术选型，带领{num}人团队完成项目交付",
    "独立负责模块设计与开发，编写技术文档{pct}份，代码评审通过率达{num}%",
    "主导技术攻关，解决系统性能瓶颈，推动项目提前{num}天上线",
    "负责跨部门协作，与产品、测试团队紧密配合，确保项目按时高质量交付",
]


def parse_skills(row):
    """解析技能字段"""
    skills = {}
    skill_fields = [
        ('编程语言', '编程语言熟练度'),
        ('前端技术', '前端技术熟练度'),
        ('后端技术', '后端技术熟练度'),
        ('数据库', '数据库熟练度'),
        ('云计算/运维', '云计算/运维熟练度'),
        ('数据与算法', '数据与算法熟练度'),
        ('移动开发', '移动开发熟练度'),
        ('测试工具', '测试工具熟练度'),
    ]

    for skill_col, level_col in skill_fields:
        if skill_col in row and row[skill_col]:
            skill_names = [s.strip() for s in str(row[skill_col]).split(',') if s.strip()]
            levels = [l.strip() for l in str(row.get(level_col, '')).split(',') if l.strip()]
            for i, skill in enumerate(skill_names):
                level = levels[i] if i < len(levels) else '了解'
                skills[skill] = level

    return skills


def get_work_years(row):
    """解析工作年限"""
    total = 0
    for field in ['小型企业工作经验', '中型企业工作经验', '大型企业工作经验']:
        val = row.get(field, '')
        if val and '年' in str(val):
            nums = re.findall(r'(\d+)', str(val))
            if nums:
                total += int(nums[0])
    return max(total, 1)


def generate_work_description(skills, work_years):
    """生成工作描述"""
    good_skills = [k for k, v in skills.items() if v in ['精通', '熟练']]
    if not good_skills:
        good_skills = list(skills.keys())[:3]

    skills_str = '、'.join(good_skills[:5])

    overview = random.choice(WORK_OVERVIEW_TEMPLATES).format(
        years=work_years,
        skills=skills_str
    )

    project_descs = []
    for skill in good_skills[:4]:
        if skill in PROJECT_TEMPLATES:
            template = random.choice(PROJECT_TEMPLATES[skill])
            desc = template.format(
                num=random.randint(10, 500),
                pct=random.randint(20, 95)
            )
            project_descs.append(desc)

    role = random.choice(PROJECT_ROLE_TEMPLATES).format(
        num=random.randint(3, 15),
        pct=random.randint(80, 99)
    )

    return overview + '。' + '。'.join(project_descs) + '。' + role + '。'


def generate_project_description(skills, work_years):
    """生成项目描述"""
    good_skills = [k for k, v in skills.items() if v in ['精通', '熟练']]
    if not good_skills:
        good_skills = list(skills.keys())[:3]

    backgrounds = [
        "电商平台的订单与支付系统重构",
        "企业级数据中台建设",
        "实时风控与反欺诈系统",
        "用户行为分析与推荐系统",
        "金融级分布式核心系统",
        "物联网设备管理平台",
        "在线教育直播系统",
        "医疗影像AI诊断平台",
    ]

    bg = random.choice(backgrounds)
    tech_stack = '、'.join(good_skills[:6])

    results = [
        f"系统可用性达到99.{random.randint(5, 9)}%",
        f"日均处理请求量达{random.randint(100, 1000)}万",
        f"用户满意度提升{random.randint(20, 50)}%",
        f"运营成本降低{random.randint(15, 40)}%",
    ]

    desc = f"项目背景：{bg}。技术栈：{tech_stack}。项目职责：负责核心模块设计与开发，参与技术方案评审，编写核心代码{random.randint(30, 80)}%。项目成果：{'；'.join(results[:3])}。"

    return desc


def enrich_csv(input_path: str):
    """增强CSV文件"""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    # 读取原文件 (尝试多种编码)
    content = None
    for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']:
        try:
            with open(input_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"使用编码: {encoding}")
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError("无法解码文件")

    f = StringIO(content)
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

    # 过滤掉已存在的新字段
    base_fieldnames = [f for f in fieldnames if f not in ['work_description', 'project_description']]
    new_fieldnames = base_fieldnames + ['work_description', 'project_description']

    # 处理每一行
    enriched_rows = []
    for row in rows:
        # 只保留基础字段
        clean_row = {k: row.get(k, '') for k in base_fieldnames}

        skills = parse_skills(clean_row)
        work_years = get_work_years(clean_row)

        clean_row['work_description'] = generate_work_description(skills, work_years)
        clean_row['project_description'] = generate_project_description(skills, work_years)
        enriched_rows.append(clean_row)

    # 写回原文件
    with open(input_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)

    print(f"[OK] 已增强 {len(enriched_rows)} 条简历数据")
    print(f"新增字段: work_description, project_description")


if __name__ == '__main__':
    enrich_csv('data/resume_data.csv')
