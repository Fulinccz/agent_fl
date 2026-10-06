import axios, { AxiosError, type AxiosInstance } from 'axios';

const baseUrl = (import.meta.env.VITE_API_URL as string) || '/api';

/**
 * 统一 HTTP 客户端
 * - 自动携带 Authorization
 * - 统一错误处理与提示
 * - 超时 120s（兼容长 LLM 推理）
 */
const http: AxiosInstance = axios.create({
  baseURL: baseUrl,
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
});

// 请求拦截器：注入 Token
http.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// 响应拦截器：统一错误处理
http.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response) {
      const status = error.response.status;
      const data = error.response.data as { message?: string; detail?: string };
      const msg = data?.message || data?.detail || `请求失败（${status}）`;
      if (status === 401) {
        localStorage.removeItem('token');
        console.error('[HTTP] 未授权，请重新登录');
      } else {
        console.error(`[HTTP] ${status}: ${msg}`);
      }
    } else if (error.code === 'ECONNABORTED') {
      console.error('[HTTP] 请求超时');
    } else {
      console.error('[HTTP] 网络错误:', error.message);
    }
    return Promise.reject(error);
  },
);

export default http;
