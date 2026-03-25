// 核心组件直接导入（本地加载）
import React, { useState, Suspense } from 'react';
import './App.css';
// 关键资源直接导入（修正路径）
import Logo from './assets/readyInClient/react.svg';

// 使用React.lazy动态导入非关键组件
const NonCriticalComponent = React.lazy(() => 
  import('./components/NonCriticalComponent')
);

function App() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showMore, setShowMore] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      // 通过 nginx 代理到 go-service（docker 容器内推荐）
      // Vite 前端运行时应使用 import.meta.env（避免浏览器报 process is not defined）
      const baseUrl = (import.meta.env.VITE_API_URL as string) || '/api';
      const res = await fetch(`${baseUrl}/agent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      const text = await res.text();
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${text || '<empty response>'}`);
      }

      let data;
      try {
        data = text ? JSON.parse(text) : null;
      } catch (err) {
        throw new Error(`Invalid JSON response: ${text}`);
      }

      if (!data || typeof data.response !== 'string') {
        throw new Error(`Unexpected response format: ${text}`);
      }

      setResponse(data.response);
    } catch (error) {
      setResponse('Error: ' + (error instanceof Error ? error.message : String(error)));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="header">
        <img src={Logo} alt="Logo" className="logo" />
        <h1>AI Agent</h1>
      </div>
      <form onSubmit={handleSubmit} className="form">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your query"
          className="input"
        />
        <button type="submit" className="button" disabled={isLoading}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
      {response && <div className="response">{response}</div>}
      
      {/* 非关键内容（按需加载） */}
      <button 
        className="more-button"
        onClick={() => setShowMore(!showMore)}
      >
        {showMore ? 'Hide Additional Features' : 'Show More Features'}
      </button>
      
      {showMore && (
        <Suspense fallback={<div className="loading">Loading features...</div>}>
          <NonCriticalComponent />
        </Suspense>
      )}
    </div>
  );
}

export default App;