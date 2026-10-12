import re
from typing import Dict, Any, List, Optional
from logger import get_logger
from .neo4j_client import Neo4jClient

logger = get_logger(__name__)


class GraphBuilder:
    """从简历数据构建知识图谱"""

    SKILL_CATEGORIES = {
        "language": ["Java", "Python", "Go", "C++", "C#", "JavaScript", "TypeScript", "PHP", "Ruby", "Swift", "Kotlin", "Rust", "Scala", "R", "MATLAB"],
        "frontend": ["React", "Vue", "Angular", "HTML", "CSS", "Sass", "Less", "Webpack", "Vite", "Next.js", "Nuxt.js", "jQuery", "Bootstrap", "Tailwind CSS"],
        "backend": ["Spring", "Spring Boot", "Django", "Flask", "FastAPI", "Express", "NestJS", "Laravel", "Rails", "ASP.NET"],
        "database": ["MySQL", "PostgreSQL", "Oracle", "SQL Server", "MongoDB", "Redis", "Elasticsearch", "ClickHouse", "TiDB", "Cassandra", "DynamoDB", "SQLite", "Neo4j"],
        "middleware": ["Kafka", "RabbitMQ", "RocketMQ", "NATS", "Pulsar", "Zookeeper", "Nginx", "HAProxy", "Envoy"],
        "bigdata": ["Hadoop", "Spark", "Flink", "Hive", "HBase", "Storm", "Kafka", "Zookeeper", "Airflow", "DolphinScheduler"],
        "cloud": ["Docker", "Kubernetes", "K8s", "AWS", "Azure", "GCP", "阿里云", "腾讯云", "华为云", "Terraform", "Jenkins", "GitLab CI", "GitHub Actions"],
        "ai": ["PyTorch", "TensorFlow", "Keras", "Scikit-learn", "Pandas", "NumPy", "OpenCV", "NLTK", "SpaCy", "Transformers", "LangChain", "LLM", "RAG"],
        "mobile": ["Android", "iOS", "Flutter", "React Native", "UniApp", "WeChat Mini Program", "小程序"],
        "test": ["Selenium", "Appium", "JUnit", "Pytest", "Jest", "Cypress", "Postman", "JMeter", "Locust"],
    }

    def __init__(self, neo4j_client: Neo4jClient = None):
        self.client = neo4j_client or Neo4jClient()

    def _detect_skill_category(self, skill_name: str) -> str:
        """检测技能所属类别"""
        skill_upper = skill_name.upper()
        for category, skills in self.SKILL_CATEGORIES.items():
            for s in skills:
                if s.upper() in skill_upper or skill_upper in s.upper():
                    return category
        return "other"

    def _extract_skills_from_text(self, text: str) -> List[Dict[str, str]]:
        """从文本中提取技能关键词"""
        if not text:
            return []

        found_skills = []
        all_skills = []
        for skills in self.SKILL_CATEGORIES.values():
            all_skills.extend(skills)
        all_skills = sorted(set(all_skills), key=len, reverse=True)

        text_upper = text.upper()
        for skill in all_skills:
            pattern = r'\b' + re.escape(skill.upper()) + r'\b'
            if re.search(pattern, text_upper):
                category = self._detect_skill_category(skill)
                found_skills.append({"name": skill, "category": category})

        return found_skills

    def build_from_resume(self, resume: Dict[str, Any]) -> Dict[str, Any]:
        """从单条简历数据构建图谱"""
        resume_id = resume.get("resume_id", "")
        if not resume_id:
            logger.warning("Resume without resume_id, skipping")
            return {"status": "skipped", "reason": "no resume_id"}

        candidate_name = resume.get("name", "")
        target_position = resume.get("target_position", "")
        degree = resume.get("degree", "")
        university_type = resume.get("university_type", "")
        age = resume.get("age", "")
        gender = resume.get("gender", "")

        work_description = resume.get("work_description", "")
        project_description = resume.get("project_description", "")
        full_text = f"{work_description} {project_description} {target_position}"

        skills = self._extract_skills_from_text(full_text)

        cypher_statements = []

        cypher_statements.append({
            "query": """
                MERGE (c:Candidate {resume_id: $resume_id})
                SET c.name = $name,
                    c.target_position = $target_position,
                    c.degree = $degree,
                    c.university_type = $university_type,
                    c.age = $age,
                    c.gender = $gender
            """,
            "params": {
                "resume_id": resume_id,
                "name": candidate_name,
                "target_position": target_position,
                "degree": degree,
                "university_type": university_type,
                "age": age,
                "gender": gender
            }
        })

        for skill in skills:
            cypher_statements.append({
                "query": """
                    MERGE (s:Skill {name: $skill_name})
                    SET s.category = $category
                    MERGE (c:Candidate {resume_id: $resume_id})
                    MERGE (c)-[r:HAS_SKILL]->(s)
                    SET r.proficiency = $proficiency
                """,
                "params": {
                    "skill_name": skill["name"],
                    "category": skill["category"],
                    "resume_id": resume_id,
                    "proficiency": "熟悉"
                }
            })

        if target_position:
            cypher_statements.append({
                "query": """
                    MERGE (p:JobPosition {name: $position_name})
                    MERGE (c:Candidate {resume_id: $resume_id})
                    MERGE (c)-[:TARGETS]->(p)
                """,
                "params": {
                    "position_name": target_position,
                    "resume_id": resume_id
                }
            })

        executed = 0
        for stmt in cypher_statements:
            try:
                self.client.run(stmt["query"], stmt["params"])
                executed += 1
            except Exception as e:
                logger.error(f"Cypher execution failed: {e}")

        logger.info(f"Built graph for resume {resume_id}: {len(skills)} skills, {executed} statements")
        return {
            "status": "success",
            "resume_id": resume_id,
            "skills_extracted": len(skills),
            "statements_executed": executed
        }

    def build_from_resumes_batch(self, resumes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量构建知识图谱"""
        total = len(resumes)
        success = 0
        failed = 0

        for resume in resumes:
            result = self.build_from_resume(resume)
            if result["status"] == "success":
                success += 1
            else:
                failed += 1

        return {
            "status": "completed",
            "total": total,
            "success": success,
            "failed": failed
        }
