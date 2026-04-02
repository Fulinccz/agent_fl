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
  const [thoughts, setThoughts] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showMore, setShowMore] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setResponse('');
    setThoughts([]); // 清空之前的思考过程
    try {
      const baseUrl = (import.meta.env.VITE_API_URL as string) || '/api';
      const res = await fetch(`${baseUrl}/agent/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      if (!res.body) throw new Error('No response body');
      const reader = res.body.getReader();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        // 解码并处理数据
        buffer += new TextDecoder().decode(value);
        
        // 分割可能的多个JSON对象
        const lines = buffer.split('\n');
        for (let i = 0; i < lines.length - 1; i++) {
          const line = lines[i].trim();
          if (line) {
            try {
              const data = JSON.parse(line);
              if (data.type === 'thought') {
                // 更新思考过程
                setThoughts(prev => [...prev, data.content]);
              } else if (data.type === 'token' && data.full_text) {
                // 更新完整结果
                setResponse(data.full_text);
              } else if (data.type === 'error') {
                // 处理错误
                setThoughts(prev => [...prev, `[错误] ${data.content}`]);
              }
            } catch (parseError) {
              console.error('Error parsing JSON:', parseError);
            }
          }
        }
        
        // 保留最后一行（可能不完整）
        buffer = lines[lines.length - 1];
      }
    } catch (error) {
      setResponse('Error: ' + (error instanceof Error ? error.message : String(error)));
      setThoughts(prev => [...prev, `[错误] ${error instanceof Error ? error.message : String(error)}`]);
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
        <div className="input-container">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your query"
            className="input"
            rows={5}
            style={{ resize: 'vertical' }}
          />
          <button type="submit" className="send-button" disabled={isLoading}>
            {isLoading ? (
              <span className="loading-icon">⏳</span>
            ) : (
              <span className="send-icon">📤</span>
            )}
          </button>
        </div>
      </form>
      <div className="output-container">
        <div className="thoughts-window">
          <div className="thoughts-content">
            {thoughts.map((thought, index) => (
              <div key={index} className="thought-item">
                [思考] {thought}
              </div>
            ))}
          </div>
        </div>
        <div className="result-window">
          <div className="response">{response || '请输入查询内容'}</div>
        </div>
      </div>
      
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