"""
职位(JD)采集主程序（并发优化版）
"""
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Any

from .crawler import JobCrawler
from .database import JobDatabase

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.logger import logger
from config.settings import collector_config


class JobCollector:
    """职位采集器（并发优化版）"""

    def __init__(self, db: JobDatabase = None):
        self.crawler = JobCrawler(max_concurrent=5)
        self.db = db or JobDatabase()

    async def collect(
        self,
        keywords: List[str],
        city: str = None,
        max_pages: int = None,
        fetch_jd: bool = True
    ) -> Dict[str, Any]:
        """
        采集职位（并发优化）

        Args:
            keywords: 关键词列表
            city: 城市
            max_pages: 最大页数
            fetch_jd: 是否获取 JD

        Returns:
            采集结果统计
        """
        max_pages = max_pages or collector_config.max_pages
        total_stats = {'total_fetched': 0, 'saved': 0, 'skipped': 0, 'failed': 0}

        for keyword in keywords:
            logger.info("开始采集关键词: %s", keyword)

            # 1. 并发搜索所有页面
            jobs = await self.crawler.search_jobs_concurrent(
                keyword=keyword,
                city=city,
                max_pages=max_pages
            )

            if not jobs:
                logger.warning("关键词 %s 未找到职位", keyword)
                continue

            logger.info("关键词 %s 找到 %d 个职位", keyword, len(jobs))

            # 2. 并发获取 JD
            if fetch_jd:
                logger.info("开始并发获取 JD...")
                jobs = await self.crawler.fetch_jd_concurrent(
                    jobs=jobs,
                    max_concurrent=10
                )

            # 3. 批量写入数据库
            db_jobs = [self._to_db_model(job) for job in jobs]
            result = self.db.save_jobs(db_jobs)

            total_stats['total_fetched'] += len(jobs)
            total_stats['saved'] += result['saved']
            total_stats['skipped'] += result['skipped']
            total_stats['failed'] += result['failed']

            logger.info("关键词 %s 处理完成: 获取 %d, 保存 %d",
                       keyword, len(jobs), result['saved'])

        return total_stats

    def _to_db_model(self, job: dict) -> dict:
        """转换为数据库模型"""
        return {
            'job_id': job.get('id', ''),
            'title': job.get('title', ''),
            'company': job.get('company'),
            'salary': job.get('salary'),
            'location': job.get('location'),
            'tags': job.get('tags', []),
            'jd': job.get('jd'),
            'source': job.get('source'),
            'source_url': job.get('source_url'),
            'raw_data': {
                'fetched_at': datetime.utcnow().isoformat()
            }
        }


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='职位采集器')
    parser.add_argument('keywords', nargs='+', help='搜索关键词')
    parser.add_argument('--city', help='城市')
    parser.add_argument('--pages', type=int, default=None, help='页数（默认读取 MAX_PAGES）')
    parser.add_argument('--no-jd', action='store_true', help='不获取 JD')
    args = parser.parse_args()

    collector = JobCollector()
    result = asyncio.run(collector.collect(
        keywords=args.keywords,
        city=args.city,
        max_pages=args.pages,
        fetch_jd=not args.no_jd
    ))

    print("\n采集完成:")
    print(f"  获取: {result['total_fetched']}")
    print(f"  保存: {result['saved']}")
    print(f"  跳过: {result['skipped']}")
    print(f"  失败: {result['failed']}")


if __name__ == '__main__':
    main()
