"""
简历数据模型（精简版）
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Resume(Base):
    """简历信息表"""
    __tablename__ = 'resumes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    resume_id = Column(String(64), unique=True, nullable=False, index=True, comment='简历编号')
    gender = Column(String(10), nullable=True, comment='性别')
    age = Column(Integer, nullable=True, comment='年龄')
    target_position = Column(String(100), nullable=True, comment='意向岗位')
    degree = Column(String(20), nullable=True, comment='学历层次')
    university_type = Column(String(50), nullable=True, comment='院校类别')
    work_description = Column(Text, nullable=True, comment='工作描述')
    project_description = Column(Text, nullable=True, comment='项目描述')
    source = Column(String(50), default='tianchi', comment='数据来源')
    created_at = Column(DateTime, default=datetime.utcnow, comment='创建时间')
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment='更新时间')

    __table_args__ = (
        Index('idx_target_position', 'target_position'),
        Index('idx_degree', 'degree'),
        Index('idx_source', 'source'),
        {'comment': '简历信息表'}
    )
