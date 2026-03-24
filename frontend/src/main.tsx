import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { monitorPerformance } from './utils/performanceMonitor' // 新增

// 启动性能监控
monitorPerformance() // 新增

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)