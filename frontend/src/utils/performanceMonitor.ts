export const monitorPerformance = () => {
  // 监控页面加载性能
  window.addEventListener('load', () => {
    const perfData = performance.timing;
    const loadTime = perfData.loadEventEnd - perfData.navigationStart;
    console.log('页面加载时间:', loadTime, 'ms');
  });

  // 监控资源加载
  const observer = new PerformanceObserver((list) => {
    list.getEntries().forEach((entry) => {
      console.log(`${entry.name} 加载时间:`, entry.duration, 'ms');
    });
  });
  observer.observe({ entryTypes: ['resource'] });
};