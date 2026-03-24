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
  // CDN 基础路径（生产环境）
  base: process.env.NODE_ENV === 'production' ? 'https://cdn.example.com/' : '/',
});