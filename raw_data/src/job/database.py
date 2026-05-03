"""
职位(JD)数据库管理
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import job_db_config
from src.logger import logger
from .models import Base, JobDescription


class JobDatabase:
    """职位数据库管理"""

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._init_engine()

    def _init_engine(self):
        """初始化数据库连接"""
        try:
            self.engine = create_engine(
                job_db_config.connection_string,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False
            )
            Base.metadata.create_all(bind=self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info("职位数据库连接初始化完成")
        except Exception as e:
            logger.error(f"职位数据库连接初始化失败: {e}")
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """获取数据库会话"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def save_jobs(self, jobs: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        保存职位列表

        Args:
            jobs: 职位字典列表

        Returns:
            统计结果
        """
        saved = 0
        skipped = 0
        failed = 0

        with self.get_session() as session:
            for job in jobs:
                try:
                    existing = session.query(JobDescription).filter_by(
                        job_id=job['job_id']
                    ).first()

                    if existing:
                        skipped += 1
                        continue

                    db_job = JobDescription(
                        job_id=job['job_id'],
                        title=job.get('title', ''),
                        company=job.get('company'),
                        salary=job.get('salary'),
                        location=job.get('location'),
                        tags=job.get('tags', []),
                        jd=job.get('jd'),
                        source=job.get('source'),
                        source_url=job.get('source_url'),
                        raw_data=job.get('raw_data', {})
                    )
                    session.add(db_job)
                    session.flush()
                    saved += 1
                    logger.info(f"保存职位: {db_job.title} @ {db_job.company}")

                except IntegrityError:
                    skipped += 1
                    session.rollback()
                except Exception as e:
                    failed += 1
                    logger.error(f"保存职位失败: {e}")
                    session.rollback()

        return {'saved': saved, 'skipped': skipped, 'failed': failed}

    def get_jobs(
        self,
        keyword: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """查询职位列表"""
        with self.get_session() as session:
            query = session.query(JobDescription)

            if keyword:
                query = query.filter(JobDescription.title.contains(keyword))
            if source:
                query = query.filter_by(source=source)

            jobs = query.order_by(JobDescription.created_at.desc()).limit(limit).offset(offset).all()
            return [self._job_to_dict(job) for job in jobs]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.get_session() as session:
            total = session.query(func.count(JobDescription.id)).scalar()
            source_stats = session.query(
                JobDescription.source,
                func.count(JobDescription.id)
            ).group_by(JobDescription.source).all()

            return {
                'total_jobs': total,
                'source_stats': {s: c for s, c in source_stats if s}
            }

    def _job_to_dict(self, job: JobDescription) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': job.id,
            'job_id': job.job_id,
            'title': job.title,
            'company': job.company,
            'salary': job.salary,
            'location': job.location,
            'tags': job.tags,
            'jd': job.jd,
            'source': job.source,
            'source_url': job.source_url,
            'created_at': job.created_at.isoformat() if job.created_at else None
        }


job_db = JobDatabase()
