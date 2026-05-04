"""
职位(JD)爬虫 - 智联招聘 + 前程无忧
基于 Playwright + BeautifulSoup 实现
优化版：并发爬取 + 浏览器复用
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
    """职位爬虫（并发优化版）"""

    def __init__(self, headless: bool = True, timeout: int = 30000, max_concurrent: int = 5):
        self.headless = headless
        self.timeout = timeout
        self.max_concurrent = max_concurrent  # 并发数限制
        self.browser = None
        self.context = None
        self.playwright = None
        self._semaphore = None  # 信号量控制并发
        logger.info("职位爬虫初始化完成 (max_concurrent=%d)", max_concurrent)

    async def _init_browser(self):
        """初始化浏览器（只启动一次）"""
        if not self.browser:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless
            )
            self.context = await self.browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
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

    async def search_jobs_concurrent(
        self,
        keyword: str,
        city: Optional[str] = None,
        max_pages: int = 5
    ) -> List[Dict[str, Any]]:
        """
        并发搜索多页职位

        Args:
            keyword: 搜索关键词
            city: 城市
            max_pages: 最大页数

        Returns:
            职位列表
        """
        try:
            await self._init_browser()

            # 并发爬取所有页面
            tasks = []
            for page in range(1, max_pages + 1):
                tasks.append(self._search_single_page(keyword, city, page))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            jobs = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error("页面爬取失败: %s", result)
                    continue
                jobs.extend(result)

            logger.info("并发搜索完成，总共找到 %d 个职位", len(jobs))
            return jobs

        except Exception as e:
            logger.error("搜索职位失败: %s", e)
            return []
        finally:
            await self._close_browser()

    async def _search_single_page(
        self,
        keyword: str,
        city: Optional[str] = None,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """搜索单页（带信号量控制并发）"""
        async with self._semaphore:
            jobs = []

            sources = [
                ('zhaopin', self._search_zhaopin),
                ('51job', self._search_51job),
            ]

            for source_name, search_func in sources:
                try:
                    logger.info("从 %s 搜索第 %d 页...", source_name, page)
                    source_jobs = await asyncio.wait_for(
                        search_func(keyword, city, page),
                        timeout=30
                    )
                    for job in source_jobs:
                        job['source'] = source_name
                    jobs.extend(source_jobs)
                    logger.info("从 %s 第 %d 页获取到 %d 个职位",
                               source_name, page, len(source_jobs))
                except asyncio.TimeoutError:
                    logger.warning("从 %s 第 %d 页获取职位超时", source_name, page)
                except Exception as e:
                    logger.error("从 %s 第 %d 页获取职位失败: %s", source_name, page, e)
                    continue

            return jobs

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
                    logger.error("解析智联招聘职位项失败: %s", e)
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
            logger.error("解析智联招聘项失败: %s", e)
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

    async def fetch_jd_concurrent(
        self,
        jobs: List[Dict[str, Any]],
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """
        并发获取 JD

        Args:
            jobs: 职位列表
            max_concurrent: 最大并发数

        Returns:
            补充 JD 后的职位列表
        """
        try:
            await self._init_browser()
            semaphore = asyncio.Semaphore(max_concurrent)

            async def fetch_single(job: Dict[str, Any]) -> Dict[str, Any]:
                async with semaphore:
                    if not job.get('source_url') or not job.get('source'):
                        return job

                    try:
                        jd = await asyncio.wait_for(
                            self._fetch_jd_single(job['source_url'], job['source']),
                            timeout=30
                        )
                        if jd:
                            job['jd'] = jd
                    except asyncio.TimeoutError:
                        logger.warning("获取 JD 超时: %s", job.get('source_url'))
                    except Exception as e:
                        logger.error("获取 JD 失败: %s", e)

                    return job

            tasks = [fetch_single(job) for job in jobs]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            processed = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error("JD 获取失败: %s", result)
                    continue
                processed.append(result)

            logger.info("JD 获取完成: %d/%d", len(processed), len(jobs))
            return processed

        except Exception as e:
            logger.error("批量获取 JD 失败: %s", e)
            return jobs
        finally:
            await self._close_browser()

    async def _fetch_jd_single(self, source_url: str, source: str) -> Optional[str]:
        """获取单个 JD"""
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

    # 兼容旧接口
    async def search_jobs(
        self,
        keyword: str,
        city: Optional[str] = None,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """兼容旧版单页搜索"""
        return await self.search_jobs_concurrent(keyword, city, max_pages=page)

    async def fetch_jd(self, source_url: str, source: str) -> Optional[str]:
        """兼容旧版单 JD 获取"""
        return await self._fetch_jd_single(source_url, source)


def run_async(coro):
    """运行异步函数"""
    return asyncio.run(coro)
