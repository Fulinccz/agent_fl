// 前端性能监控：仅开发环境启用，避免生产控制台刷屏；
// observer 在页面加载 5s 后断开，防止长会话无界收集。
export const monitorPerformance = () => {
  if (!import.meta.env.DEV) return;

  window.addEventListener('load', () => {
    // performance.timing 已废弃，使用 Navigation Timing Level 2
    const nav = performance.getEntriesByType('navigation')[0] as
      | PerformanceNavigationTiming
      | undefined;
    if (nav) {
      console.log('页面加载时间:', Math.round(nav.loadEventEnd - nav.startTime), 'ms');
    }
  });

  const observer = new PerformanceObserver((list) => {
    list.getEntries().forEach((entry) => {
      console.log(`${entry.name} 加载时间:`, Math.round(entry.duration), 'ms');
    });
  });
  observer.observe({ entryTypes: ['resource'] });

  window.addEventListener('load', () => {
    setTimeout(() => observer.disconnect(), 5000);
  });
};
