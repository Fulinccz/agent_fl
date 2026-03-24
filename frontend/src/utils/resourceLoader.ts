// CDN 基础 URL
const CDN_BASE_URL = import.meta.env.PROD 
  ? 'https://cdn.example.com/' 
  : '/';

/**
 * 加载图片资源
 * @param path 资源路径
 * @param isCritical 是否为关键资源
 */
export const loadImage = (path: string, isCritical = false): string => {
  if (isCritical) {
    return path;
  }
  return `${CDN_BASE_URL}${path}`;
};

/**
 * 预加载资源
 * @param urls 资源 URL 数组
 */
export const preloadResources = (urls: string[]): void => {
  urls.forEach(url => {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = url;
    link.as = 'image';
    document.head.appendChild(link);
  });
};