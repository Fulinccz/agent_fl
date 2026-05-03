"""
简历数据导入器（精简版）
直接全部导入，不做关键词筛选
"""
import csv
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.logger import logger
from .database import ResumeDatabase
from config.settings import resume_config


class ResumeImporter:
    """简历数据导入器"""

    def __init__(self, db: Optional[ResumeDatabase] = None):
        self.db = db or ResumeDatabase()

    def import_from_csv(
        self,
        csv_path: str,
        batch_size: int = None,
        max_batches: int = None
    ) -> Dict[str, int]:
        """
        从 CSV 文件导入简历数据（全部导入，不做筛选）

        Args:
            csv_path: CSV 文件路径
            batch_size: 批量插入大小，默认读取 RESUME_BATCH_SIZE
            max_batches: 最大批次数，默认读取 RESUME_MAX_BATCHES

        Returns:
            导入统计
        """
        batch_size = batch_size or resume_config.batch_size
        max_batches = max_batches or resume_config.max_batches

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

        logger.info(f"开始导入简历数据: {csv_path}")
        logger.info(f"批量大小: {batch_size}, 最大批次数: {max_batches}")

        total_saved = 0
        total_skipped = 0
        total_failed = 0
        batch_count = 0
        batch = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # 达到最大批次数，结束
                if batch_count >= max_batches:
                    logger.info(f"达到最大批次数 {max_batches}，停止导入")
                    break

                try:
                    resume = self._parse_row(row)
                    if not resume:
                        continue

                    batch.append(resume)

                    # 满一批就写入
                    if len(batch) >= batch_size:
                        result = self.db.save_resumes(batch)
                        batch_count += 1
                        total_saved += result['saved']
                        total_skipped += result['skipped']
                        total_failed += result['failed']
                        logger.info(f"第 {batch_count} 批写入完成: +{result['saved']} 条"
                                    f" (累计: {total_saved} 条)")
                        batch = []

                except Exception as e:
                    logger.error(f"解析行失败: {e}")
                    total_failed += 1

            # 处理剩余批次（未达最大批次数时）
            if batch and batch_count < max_batches:
                result = self.db.save_resumes(batch)
                batch_count += 1
                total_saved += result['saved']
                total_skipped += result['skipped']
                total_failed += result['failed']
                logger.info(f"第 {batch_count} 批写入完成: +{result['saved']} 条"
                            f" (累计: {total_saved} 条)")

        logger.info(f"导入完成: 成功 {total_saved}, 跳过 {total_skipped}, 失败 {total_failed}")
        return {
            'saved': total_saved,
            'skipped': total_skipped,
            'failed': total_failed,
            'batches': batch_count
        }

    def _parse_row(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        解析 CSV 行数据（只提取需要的字段）
        """
        try:
            name = row.get('姓名', row.get('name', ''))
            resume = {
                'resume_id': name or f"resume_{hash(str(row))}",
                'gender': row.get('性别', row.get('gender')),
                'age': self._parse_int(row.get('年龄', row.get('age'))),
                'target_position': row.get('意向岗位', row.get('target_position')),
                'degree': row.get('学历层次', row.get('degree')),
                'university_type': row.get('院校类别', row.get('university_type')),
                'work_description': row.get('work_description', ''),
                'project_description': row.get('project_description', ''),
                'source': 'tianchi',
            }

            if not resume['resume_id']:
                return None

            return resume

        except Exception as e:
            logger.error(f"解析行数据失败: {e}")
            return None

    def _parse_int(self, value: Any) -> Optional[int]:
        """解析整数"""
        if not value:
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='导入简历数据')
    parser.add_argument('csv_file', help='CSV 文件路径')
    parser.add_argument('--batch-size', type=int, default=None, help='批量大小')
    parser.add_argument('--max-batches', type=int, default=None, help='最大批次数')
    args = parser.parse_args()

    importer = ResumeImporter()
    result = importer.import_from_csv(
        args.csv_file,
        batch_size=args.batch_size,
        max_batches=args.max_batches
    )
    print(f"\n导入结果:")
    print(f"  成功: {result['saved']}")
    print(f"  跳过: {result['skipped']}")
    print(f"  失败: {result['failed']}")
    print(f"  批次数: {result['batches']}")


if __name__ == '__main__':
    main()
