"""
简历数据库管理（精简版）
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
from config.settings import resume_db_config
from src.logger import logger
from .models import Base, Resume


class ResumeDatabase:
    """简历数据库管理"""

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._init_engine()

    def _init_engine(self):
        """初始化数据库连接"""
        try:
            self.engine = create_engine(
                resume_db_config.connection_string,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False
            )
            Base.metadata.create_all(bind=self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info("简历数据库连接初始化完成")
        except Exception as e:
            logger.error(f"简历数据库连接初始化失败: {e}")
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

    def save_resumes(self, resumes: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        批量保存简历

        Args:
            resumes: 简历字典列表

        Returns:
            统计结果
        """
        saved = 0
        skipped = 0
        failed = 0

        with self.get_session() as session:
            for resume in resumes:
                try:
                    existing = session.query(Resume).filter_by(
                        resume_id=resume['resume_id']
                    ).first()

                    if existing:
                        skipped += 1
                        continue

                    db_resume = Resume(
                        resume_id=resume['resume_id'],
                        gender=resume.get('gender'),
                        age=resume.get('age'),
                        target_position=resume.get('target_position'),
                        degree=resume.get('degree'),
                        university_type=resume.get('university_type'),
                        work_description=resume.get('work_description'),
                        project_description=resume.get('project_description'),
                        source=resume.get('source', 'tianchi')
                    )
                    session.add(db_resume)
                    session.flush()
                    saved += 1

                except IntegrityError:
                    skipped += 1
                    session.rollback()
                except Exception as e:
                    failed += 1
                    logger.error(f"保存简历失败: {e}")
                    session.rollback()

        return {'saved': saved, 'skipped': skipped, 'failed': failed}

    def get_resumes(
        self,
        target_position: Optional[str] = None,
        degree: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """查询简历列表"""
        with self.get_session() as session:
            query = session.query(Resume)

            if target_position:
                query = query.filter(Resume.target_position.contains(target_position))
            if degree:
                query = query.filter_by(degree=degree)

            resumes = query.order_by(Resume.created_at.desc()).limit(limit).offset(offset).all()
            return [self._resume_to_dict(r) for r in resumes]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.get_session() as session:
            total = session.query(func.count(Resume.id)).scalar()
            position_stats = session.query(
                Resume.target_position,
                func.count(Resume.id)
            ).group_by(Resume.target_position).all()

            degree_stats = session.query(
                Resume.degree,
                func.count(Resume.id)
            ).group_by(Resume.degree).all()

            return {
                'total_resumes': total,
                'position_stats': {p: c for p, c in position_stats if p},
                'degree_stats': {d: c for d, c in degree_stats if d}
            }

    def _resume_to_dict(self, resume: Resume) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': resume.id,
            'resume_id': resume.resume_id,
            'gender': resume.gender,
            'age': resume.age,
            'target_position': resume.target_position,
            'degree': resume.degree,
            'university_type': resume.university_type,
            'work_description': resume.work_description,
            'project_description': resume.project_description,
            'source': resume.source,
            'created_at': resume.created_at.isoformat() if resume.created_at else None
        }


resume_db = ResumeDatabase()
