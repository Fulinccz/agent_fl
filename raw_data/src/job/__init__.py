"""
职位(JD)采集模块
"""
from .crawler import JobCrawler
from .database import JobDatabase

__all__ = ['JobCrawler', 'JobDatabase']
