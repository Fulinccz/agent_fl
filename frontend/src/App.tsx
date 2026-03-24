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
      const res = await fetch('http://localhost:8080/api/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      const data = await res.json();
      setResponse(data.response);
    } catch (error) {
      setResponse('Error: ' + error);
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