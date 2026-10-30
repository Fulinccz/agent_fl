import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      // 本地开发：将 /api 转发到后端网关
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        secure: false,
        // 本地 GGUF 推理慢：面试攻略全量生成单次可达 5 分钟以上，
        // 代理超时必须留足余量，否则 SSE 会在生成完成前被截断
        timeout: 900000,
        proxyTimeout: 900000,
      },
    },
  },
  build: {
    // 分离CSS
    cssCodeSplit: true,
    // 优化资源输出
    rollupOptions: {
      output: {
        // 分离第三方库
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
        },
        // 资源命名规则
        assetFileNames: 'assets/[name].[hash].[ext]',
        chunkFileNames: 'chunks/[name].[hash].js',
        entryFileNames: 'entry/[name].[hash].js',
      },
    },
  },
  // 基础路径
  base: '/',
});