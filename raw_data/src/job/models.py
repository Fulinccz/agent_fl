"""
职位(JD)数据模型
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class JobDescription(Base):
    """职位信息表"""
    __tablename__ = 'job_descriptions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), unique=True, nullable=False, index=True, comment='职位ID')
    title = Column(String(255), nullable=False, comment='职位标题')
    company = Column(String(255), nullable=True, comment='公司名称')
    salary = Column(String(100), nullable=True, comment='薪资文本')
    location = Column(String(255), nullable=True, comment='工作地点')
    tags = Column(JSON, default=list, comment='标签列表')
    jd = Column(Text, nullable=True, comment='职位描述')
    source = Column(String(50), nullable=True, comment='数据来源')
    source_url = Column(String(500), nullable=True, comment='原始链接')
    raw_data = Column(JSON, default=dict, comment='原始数据')
    created_at = Column(DateTime, default=datetime.utcnow, comment='创建时间')
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment='更新时间')

    __table_args__ = (
        Index('idx_title', 'title'),
        Index('idx_company', 'company'),
        Index('idx_source', 'source'),
        Index('idx_created_at', 'created_at'),
        {'comment': '职位信息表'}
    )
