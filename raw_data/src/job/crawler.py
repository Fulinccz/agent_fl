"""
职位(JD)爬虫 - 智联招聘 + 前程无忧
基于 Playwright + BeautifulSoup 实现
"""

import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from urllib.parse import quote

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.logger import logger


class JobCrawler:
    """职位爬虫"""

    def __init__(self, headless: bool = True, timeout: int = 30000):
        self.headless = headless
        self.timeout = timeout
        self.browser = None
        self.context = None
        self.playwright = None
        logger.info("职位爬虫初始化完成")

    async def _init_browser(self):
        """初始化浏览器"""
        if not self.browser:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless
            )
            self.context = await self.browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            logger.info("浏览器启动完成")

    async def _close_browser(self):
        """关闭浏览器"""
        if self.context:
            await self.context.close()
            self.context = None
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        logger.info("浏览器已关闭")

    async def search_jobs(
        self,
        keyword: str,
        city: Optional[str] = None,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """
        搜索职位

        Args:
            keyword: 搜索关键词
            city: 城市
            page: 页码

        Returns:
            职位列表
        """
        try:
            await self._init_browser()

            jobs = []

            # 只保留能获取 JD 的网站
            sources = [
                ('zhaopin', self._search_zhaopin),
                ('51job', self._search_51job),
            ]

            for source_name, search_func in sources:
                try:
                    logger.info(f"从 {source_name} 搜索职位...")
                    source_jobs = await asyncio.wait_for(
                        search_func(keyword, city, page),
                        timeout=30
                    )
                    for job in source_jobs:
                        job['source'] = source_name
                    jobs.extend(source_jobs)
                    logger.info(f"从 {source_name} 获取到 {len(source_jobs)} 个职位")
                except asyncio.TimeoutError:
                    logger.warning(f"从 {source_name} 获取职位超时")
                except Exception as e:
                    logger.error(f"从 {source_name} 获取职位失败: {e}")
                    continue

            logger.info(f"搜索完成，总共找到 {len(jobs)} 个职位")
            return jobs

        except Exception as e:
            logger.error(f"搜索职位失败: {e}")
            return []
        finally:
            await self._close_browser()

    async def _search_zhaopin(
        self,
        keyword: str,
        city: Optional[str] = None,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """搜索智联招聘"""
        search_keyword = f"{keyword} {city}" if city else keyword
        url = f"https://sou.zhaopin.com/?jl=489&kw={quote(search_keyword)}&p={page}"

        page_obj = await self.context.new_page()
        try:
            await page_obj.goto(url, wait_until='networkidle', timeout=self.timeout)
            await asyncio.sleep(2)

            content = await page_obj.content()
            soup = BeautifulSoup(content, 'html.parser')

            jobs = []
            job_items = soup.select('.joblist-box__item')

            for item in job_items:
                try:
                    job = self._parse_zhaopin_item(item)
                    if job:
                        jobs.append(job)
                except Exception as e:
                    logger.error(f"解析智联招聘职位项失败: {e}")
                    continue

            return jobs
        finally:
            await page_obj.close()

    def _parse_zhaopin_item(self, item) -> Optional[Dict[str, Any]]:
        """解析智联招聘职位项"""
        try:
            title = item.select_one('.jobinfo__name')
            salary = item.select_one('.jobinfo__salary')
            company = item.select_one('.companyinfo__name')
            location = item.select_one('.jobinfo__other-info-item span')
            link = item.select_one('a.jobinfo__name')
            tags = item.select('.joblist-box__item-tag')

            # 生成短 ID
            url = link.get('href', '') if link else ''
            short_id = hashlib.md5(url.encode()).hexdigest()[:16]

            return {
                'id': short_id,
                'title': title.text.strip() if title else '',
                'salary': salary.text.strip() if salary else '',
                'company': company.text.strip() if company else '',
                'location': location.text.strip() if location else '',
                'tags': [t.text.strip() for t in tags],
                'source_url': url,
            }
        except Exception as e:
            logger.error(f"解析智联招聘项失败: {e}")
            return None

    async def _search_51job(
        self,
        keyword: str,
        city: Optional[str] = None,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """搜索前程无忧"""
        search_keyword = f"{keyword} {city}" if city else keyword
        url = f"https://we.51job.com/pc/search?keyword={quote(search_keyword)}&searchType=2&pageNum={page}"

        page_obj = await self.context.new_page()
        try:
            await page_obj.goto(url, wait_until='networkidle', timeout=self.timeout)
            await page_obj.wait_for_selector('.joblist-item', timeout=10000)
            await asyncio.sleep(3)

            # 使用 page.evaluate 在浏览器中执行提取（前程无忧是动态渲染）
            jobs = await page_obj.evaluate("""
                () => {
                    const items = document.querySelectorAll('.joblist-item');
                    const result = [];
                    items.forEach(item => {
                        const jobWrapper = item.querySelector('.joblist-item-job');
                        let sensorsData = {};
                        if (jobWrapper) {
                            try {
                                sensorsData = JSON.parse(jobWrapper.getAttribute('sensorsdata') || '{}');
                            } catch(e) {}
                        }

                        const titleEl = item.querySelector('.jname, .joblist-item-jobname span');
                        const title = titleEl?.textContent?.trim() || sensorsData.jobTitle || '';

                        let salary = sensorsData.jobSalary || '';
                        if (!salary) {
                            const salaryEl = item.querySelector('.sal, [class*="salary"]');
                            salary = salaryEl?.textContent?.trim() || '';
                        }

                        let company = '';
                        const companyEl = item.querySelector('.joblist-item-right .cname, .comp-name, [class*="company"]');
                        company = companyEl?.textContent?.trim() || '';

                        let location = sensorsData.jobArea || '';
                        if (!location) {
                            const locEl = item.querySelector('.area, [class*="area"]');
                            location = locEl?.textContent?.trim() || '';
                        }

                        const link = item.querySelector('a');
                        const href = link?.href || '';

                        const tags = [];
                        item.querySelectorAll('.tag').forEach(t => {
                            tags.push(t.textContent?.trim());
                        });

                        const shortId = href ? btoa(href).substring(0, 16) : '';

                        result.push({
                            id: shortId,
                            title: title,
                            salary: salary,
                            company: company,
                            location: location,
                            tags: tags.filter(Boolean),
                            source_url: href
                        });
                    });
                    return result;
                }
            """)

            return jobs
        finally:
            await page_obj.close()

    async def fetch_jd(self, source_url: str, source: str) -> Optional[str]:
        """
        获取职位描述（JD）

        Args:
            source_url: 职位详情页 URL
            source: 数据来源

        Returns:
            JD 文本或 None
        """
        try:
            await self._init_browser()
            page_obj = await self.context.new_page()

            try:
                await page_obj.goto(source_url, wait_until='networkidle', timeout=self.timeout)
                await asyncio.sleep(2)

                content = await page_obj.content()
                soup = BeautifulSoup(content, 'html.parser')

                jd = None

                if source == 'zhaopin':
                    jd_elem = soup.select_one('.describtion__detail-content') or soup.select_one('[class*="detail"]')
                    if jd_elem:
                        jd = jd_elem.get_text(strip=True, separator='\n')

                elif source == '51job':
                    jd_elem = soup.select_one('.job-desc') or soup.select_one('[class*="detail"]')
                    if jd_elem:
                        jd = jd_elem.get_text(strip=True, separator='\n')

                if jd and len(jd) > 50:
                    return jd[:5000]
                return None

            finally:
                await page_obj.close()

        except Exception as e:
            logger.error(f"获取 JD 失败: {e}")
            return None
        finally:
            await self._close_browser()


def run_async(coro):
    """运行异步函数"""
    return asyncio.run(coro)
