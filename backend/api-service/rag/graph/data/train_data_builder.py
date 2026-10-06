import json
import re
from typing import Dict, Any, List, Optional
from logger import get_logger
from ..neo4j.neo4j_client import Neo4jClient

logger = get_logger(__name__)


class TrainDataBuilder:
    """从 row_train.json 训练数据构建知识图谱

    数据结构:
    {
        "input": "简历：... / JD：...\n简历：...",
        "output": "技术栈：...\n技能：...\n项目简介：..."
    }

    抽取实体:
    - Skill: 技术栈中的工具/框架/语言
    - Ability: 技能标签
    - Scene: 业务场景/岗位方向
    - TrainSample: 训练样本本身

    关系:
    - (TrainSample)-[:HAS_TECH_STACK]->(Skill)
    - (TrainSample)-[:HAS_ABILITY]->(Ability)
    - (TrainSample)-[:BELONGS_TO_SCENE]->(Scene)
    - (Skill)-[:USED_IN]->(Scene)
    - (Ability)-[:REQUIRES]->(Skill)
    """

    SKILL_CATEGORIES = {
        "language": ["Java", "Python", "Go", "C++", "C#", "JavaScript", "TypeScript", "PHP", "Ruby", "Swift", "Kotlin", "Rust", "Scala", "R", "MATLAB", "Shell", "SQL"],
        "frontend": ["React", "Vue", "Angular", "HTML", "CSS", "Sass", "Less", "Webpack", "Vite", "Next.js", "Nuxt.js", "jQuery", "Bootstrap", "Tailwind CSS", "微信小程序", "小程序"],
        "backend": ["Spring", "Spring Boot", "SpringBoot", "Django", "Flask", "FastAPI", "Express", "NestJS", "Laravel", "Rails", "ASP.NET", "Gin", "RESTful", "RESTful API"],
        "database": ["MySQL", "PostgreSQL", "Oracle", "SQL Server", "MongoDB", "Redis", "Elasticsearch", "ClickHouse", "TiDB", "Cassandra", "DynamoDB", "SQLite", "Neo4j", "Hive"],
        "middleware": ["Kafka", "RabbitMQ", "RocketMQ", "NATS", "Pulsar", "Zookeeper", "Nginx", "HAProxy", "Envoy"],
        "bigdata": ["Hadoop", "Spark", "Flink", "Hive", "HBase", "Storm", "Airflow", "DolphinScheduler", "DataX", "ETL"],
        "cloud": ["Docker", "Kubernetes", "K8s", "AWS", "Azure", "GCP", "阿里云", "腾讯云", "华为云", "Terraform", "Jenkins", "GitLab CI", "GitHub Actions", "CI/CD", "Uvicorn"],
        "ai": ["PyTorch", "TensorFlow", "Keras", "Scikit-learn", "Pandas", "NumPy", "OpenCV", "NLTK", "SpaCy", "Transformers", "LangChain", "LLM", "RAG", "FAISS", "Chroma", "LoRA", "bitsandbytes", "模型量化", "知识蒸馏", "KV Cache"],
        "mobile": ["Android", "iOS", "Flutter", "React Native", "UniApp", "WeChat Mini Program", "小程序"],
        "test": ["Selenium", "Appium", "JUnit", "Pytest", "Jest", "Cypress", "Postman", "JMeter", "Locust", "BurpSuite", "Nessus", "Mock"],
        "monitor": ["ELK", "Grafana", "Prometheus", "监控告警", "日志追踪"],
        "security": ["数据脱敏", "权限管控", "数据加密", "数据合规", "漏洞扫描", "渗透测试", "安全加固"],
        "ops": ["Linux", "Shell", "Nginx", "服务器运维", "环境部署", "资源优化"],
        "dataviz": ["ECharts", "数据可视化", "可视化大屏", "报表设计"],
        "other": ["Git", "GitLab", "埋点", "AB实验", "AB测试", "用户画像", "推荐系统", "CTR", "CVR", "NLP", "文本分类", "信息抽取", "语义理解", "PDF解析", "Word解析", "信息抽取", "STAR法则"]
    }

    SCENE_KEYWORDS = {
        "后端开发": ["后端", "后端开发", "后端服务", "接口开发", "业务开发", "API网关", "微服务", "高并发"],
        "前端开发": ["前端", "前端开发", "React", "Vue", "Angular", "页面开发", "组件封装"],
        "数据开发": ["数据开发", "ETL", "数仓", "数据仓库", "离线数据", "实时数据", "数据处理", "数据同步"],
        "算法工程": ["算法", "模型训练", "特征工程", "推荐系统", "广告算法", "CTR", "CVR", "用户画像"],
        "AI大模型": ["LLM", "大模型", "LangChain", "RAG", "Agent", "模型部署", "模型量化", "模型微调", "推理优化"],
        "测试开发": ["测试", "自动化测试", "性能测试", "安全测试", "CI/CD", "Mock", "压测"],
        "运维开发": ["运维", "Linux", "Shell", "监控", "日志", "故障排查", "应急响应", "部署"],
        "全栈开发": ["全栈", "前后端", "全链路"],
        "移动端开发": ["Android", "iOS", "Flutter", "React Native", "移动端", "小程序"],
        "数据可视化": ["可视化", "ECharts", "数据大屏", "报表"],
        "项目管理": ["项目管理", "需求分析", "进度把控", "团队协作", "项目交付"],
        "数据质量": ["数据质量", "数据校验", "特征校验", "数据治理"],
    }

    def __init__(self, neo4j_client: Neo4jClient = None):
        self.client = neo4j_client or Neo4jClient()

    def _detect_skill_category(self, skill_name: str) -> str:
        skill_upper = skill_name.upper()
        for category, skills in self.SKILL_CATEGORIES.items():
            for s in skills:
                if s.upper() == skill_upper or skill_upper in s.upper() or s.upper() in skill_upper:
                    return category
        return "other"

    def _extract_skills_from_text(self, text: str) -> List[Dict[str, str]]:
        if not text:
            return []

        found_skills = []
        all_skills = []
        for skills in self.SKILL_CATEGORIES.values():
            all_skills.extend(skills)
        all_skills = sorted(set(all_skills), key=len, reverse=True)

        text_upper = text.upper()
        matched_positions = set()

        for skill in all_skills:
            pattern = r'(?<![A-Za-z0-9_])' + re.escape(skill.upper()) + r'(?![A-Za-z0-9_])'
            for match in re.finditer(pattern, text_upper):
                start, end = match.span()
                if not any(start < p < end for p in matched_positions):
                    category = self._detect_skill_category(skill)
                    found_skills.append({"name": skill, "category": category})
                    matched_positions.update(range(start, end))

        return found_skills

    def _extract_abilities_from_output(self, output_text: str) -> List[str]:
        abilities = []
        if "技能：" in output_text:
            skill_section = output_text.split("技能：")[1].split("\n")[0]
            abilities = [a.strip() for a in skill_section.split("、") if a.strip()]
        return abilities

    def _extract_tech_stack_from_output(self, output_text: str) -> List[str]:
        tech_stack = []
        if "技术栈：" in output_text:
            tech_section = output_text.split("技术栈：")[1].split("\n")[0]
            tech_stack = [t.strip() for t in tech_section.split("、") if t.strip()]
        return tech_stack

    def _detect_scene(self, input_text: str, output_text: str) -> str:
        full_text = f"{input_text} {output_text}"
        for scene, keywords in self.SCENE_KEYWORDS.items():
            for kw in keywords:
                if kw in full_text:
                    return scene
        return "通用"

    def _extract_project_summary(self, output_text: str) -> str:
        if "项目简介：" in output_text:
            return output_text.split("项目简介：")[1].strip()
        return ""

    def build_from_train_sample(self, sample: Dict[str, str], sample_id: str) -> Dict[str, Any]:
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")

        if not input_text or not output_text:
            return {"status": "skipped", "reason": "empty input or output", "sample_id": sample_id}

        tech_stack = self._extract_tech_stack_from_output(output_text)
        abilities = self._extract_abilities_from_output(output_text)
        skills = self._extract_skills_from_text(f"{input_text} {output_text}")
        scene = self._detect_scene(input_text, output_text)
        project_summary = self._extract_project_summary(output_text)

        cypher_statements = []

        cypher_statements.append({
            "query": """
                MERGE (t:TrainSample {sample_id: $sample_id})
                SET t.input_text = $input_text,
                    t.output_text = $output_text,
                    t.project_summary = $project_summary,
                    t.scene = $scene
            """,
            "params": {
                "sample_id": sample_id,
                "input_text": input_text[:500],
                "output_text": output_text[:500],
                "project_summary": project_summary[:300],
                "scene": scene
            }
        })

        for skill in skills:
            cypher_statements.append({
                "query": """
                    MERGE (s:Skill {name: $skill_name})
                    SET s.category = $category
                    MERGE (t:TrainSample {sample_id: $sample_id})
                    MERGE (t)-[r:HAS_SKILL]->(s)
                    SET r.source = 'auto_extract'
                """,
                "params": {
                    "skill_name": skill["name"],
                    "category": skill["category"],
                    "sample_id": sample_id
                }
            })

        for tech in tech_stack:
            category = self._detect_skill_category(tech)
            cypher_statements.append({
                "query": """
                    MERGE (s:Skill {name: $skill_name})
                    SET s.category = $category, s.is_tech_stack = true
                    MERGE (t:TrainSample {sample_id: $sample_id})
                    MERGE (t)-[r:HAS_TECH_STACK]->(s)
                    SET r.source = 'tech_stack'
                """,
                "params": {
                    "skill_name": tech,
                    "category": category,
                    "sample_id": sample_id
                }
            })

        for ability in abilities:
            cypher_statements.append({
                "query": """
                    MERGE (a:Ability {name: $ability_name})
                    MERGE (t:TrainSample {sample_id: $sample_id})
                    MERGE (t)-[:HAS_ABILITY]->(a)
                """,
                "params": {
                    "ability_name": ability,
                    "sample_id": sample_id
                }
            })

        cypher_statements.append({
            "query": """
                MERGE (sc:Scene {name: $scene_name})
                MERGE (t:TrainSample {sample_id: $sample_id})
                MERGE (t)-[:BELONGS_TO_SCENE]->(sc)
            """,
            "params": {
                "scene_name": scene,
                "sample_id": sample_id
            }
        })

        executed = 0
        for stmt in cypher_statements:
            try:
                self.client.run(stmt["query"], stmt["params"])
                executed += 1
            except Exception as e:
                logger.error(f"Cypher execution failed for {sample_id}: {e}")

        logger.info(
            f"Built graph for sample {sample_id}: "
            f"scene={scene}, skills={len(skills)}, tech_stack={len(tech_stack)}, abilities={len(abilities)}"
        )

        return {
            "status": "success",
            "sample_id": sample_id,
            "scene": scene,
            "skills_extracted": len(skills),
            "tech_stack": len(tech_stack),
            "abilities": len(abilities),
            "statements_executed": executed
        }

    def build_from_json_file(self, json_path: str, limit: Optional[int] = None) -> Dict[str, Any]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if limit:
            data = data[:limit]

        total = len(data)
        success = 0
        failed = 0
        scene_counts = {}

        for idx, sample in enumerate(data):
            sample_id = f"TRAIN_{idx:04d}"
            result = self.build_from_train_sample(sample, sample_id)

            if result["status"] == "success":
                success += 1
                scene = result.get("scene", "未知")
                scene_counts[scene] = scene_counts.get(scene, 0) + 1
            else:
                failed += 1

            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx + 1}/{total} samples processed")

        logger.info(f"Batch build completed: total={total}, success={success}, failed={failed}")
        logger.info(f"Scene distribution: {scene_counts}")

        return {
            "status": "completed",
            "total": total,
            "success": success,
            "failed": failed,
            "scene_distribution": scene_counts
        }

    def build_skill_relationships(self):
        """构建技能共现关系: 经常一起出现的技能建立关联"""
        logger.info("Building skill co-occurrence relationships...")

        result = self.client.run("""
            MATCH (t:TrainSample)-[:HAS_SKILL]->(s1:Skill)
            MATCH (t)-[:HAS_SKILL]->(s2:Skill)
            WHERE s1.name < s2.name
            WITH s1, s2, count(t) AS cooccurrence
            WHERE cooccurrence >= 2
            MERGE (s1)-[r:CO_OCCURS_WITH]->(s2)
            SET r.weight = cooccurrence
            RETURN s1.name, s2.name, cooccurrence
            ORDER BY cooccurrence DESC
            LIMIT 20
        """)

        logger.info(f"Created {len(result)} skill co-occurrence relationships")
        return result

    def build_scene_skill_mapping(self):
        """构建场景-技能映射: 每个场景下高频技能"""
        logger.info("Building scene-skill mapping...")

        result = self.client.run("""
            MATCH (sc:Scene)<-[:BELONGS_TO_SCENE]-(t:TrainSample)-[:HAS_SKILL]->(s:Skill)
            WITH sc, s, count(t) AS freq
            WHERE freq >= 2
            MERGE (sc)-[r:REQUIRES_SKILL]->(s)
            SET r.frequency = freq
            RETURN sc.name, s.name, freq
            ORDER BY freq DESC
            LIMIT 30
        """)

        logger.info(f"Created {len(result)} scene-skill mapping relationships")
        return result
