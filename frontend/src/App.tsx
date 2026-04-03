// 核心组件直接导入（本地加载）
import React, { useState, Suspense, useEffect } from 'react';
import './App.css';
// 关键资源直接导入（修正路径）
import Logo from './assets/readyInClient/react.svg';

// 使用React.lazy动态导入非关键组件
const NonCriticalComponent = React.lazy(() => 
  import('./components/NonCriticalComponent')
);

// 历史记录项类型
interface HistoryItem {
  id: string;
  query: string;
  response: string;
  timestamp: number;
  file?: string;
}

function App() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [thoughts, setThoughts] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showMore, setShowMore] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  // 从本地存储加载历史记录
  useEffect(() => {
    const savedHistory = localStorage.getItem('chatHistory');
    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory));
      } catch (error) {
        console.error('Error loading history:', error);
      }
    }
  }, []);

  // 保存历史记录到本地存储
  useEffect(() => {
    localStorage.setItem('chatHistory', JSON.stringify(history));
  }, [history]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setUploadedFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setResponse('');
    setThoughts([]); // 清空之前的思考过程
    try {
      const baseUrl = (import.meta.env.VITE_API_URL as string) || '/api';
      let finalResponse = '';
      
      if (uploadedFile) {
        // 处理文件上传
        const formData = new FormData();
        formData.append('file', uploadedFile);
        formData.append('query', query);
        formData.append('provider', 'local');
        
        const res = await fetch(`${baseUrl}/agent/upload`, {
          method: 'POST',
          body: formData
        });
        
        if (!res.ok) throw new Error('File upload failed');
        
        const data = await res.json();
        finalResponse = data.response;
        setResponse(data.response);
      } else {
        // 处理普通文本查询
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
                  finalResponse = data.full_text;
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
      }
      
      // 保存历史记录
      if (finalResponse) {
        const newHistoryItem: HistoryItem = {
          id: Date.now().toString(),
          query,
          response: finalResponse,
          timestamp: Date.now(),
          file: uploadedFile?.name
        };
        
        setHistory(prev => [newHistoryItem, ...prev].slice(0, 50)); // 只保留最近50条记录
      }
    } catch (error) {
      const errorMessage = 'Error: ' + (error instanceof Error ? error.message : String(error));
      setResponse(errorMessage);
      setThoughts(prev => [...prev, `[错误] ${error instanceof Error ? error.message : String(error)}`]);
    } finally {
      setIsLoading(false);
      setUploadedFile(null); // 上传完成后清空文件
    }
  };

  // 格式化时间
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  // 加载历史记录项
  const loadHistoryItem = (item: HistoryItem) => {
    setQuery(item.query);
    setResponse(item.response);
    setShowHistory(false);
  };

  return (
    <div className="app">
      <div className="header">
        <img src={Logo} alt="Logo" className="logo" />
        <h1>Fulin AI</h1>
        <div className="header-actions">
          <button 
            className="history-button"
            onClick={() => setShowHistory(!showHistory)}
          >
            {showHistory ? '隐藏历史记录' : '查看历史记录'}
          </button>
        </div>
      </div>
      
      {/* 历史记录列表 */}
      {showHistory && (
        <div className="history-container">
          <h3>历史记录</h3>
          {history.length === 0 ? (
            <p className="no-history">暂无历史记录</p>
          ) : (
            <div className="history-list">
              {history.map((item) => (
                <div key={item.id} className="history-item" onClick={() => loadHistoryItem(item)}>
                  <div className="history-query">{item.query}</div>
                  {item.file && (
                    <div className="history-file">📎 {item.file}</div>
                  )}
                  <div className="history-time">{formatTime(item.timestamp)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="form">
        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                // 自动调整高度
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
              }}
              placeholder="请发送消息~"
              className="input"
              rows={1}
              style={{ resize: 'none' }}
            />
            {uploadedFile && (
              <div className="file-info">
                <span className="file-name">{uploadedFile.name}</span>
                <button 
                  type="button" 
                  className="remove-file"
                  onClick={() => setUploadedFile(null)}
                >
                  ×
                </button>
              </div>
            )}
          </div>
          <div className="form-actions">
            <div className="attachment-container">
              <input 
                type="file" 
                id="file-upload" 
                className="file-upload"
                onChange={handleFileChange}
              />
              <label htmlFor="file-upload" className="attachment-button">
                📎
              </label>
            </div>
            <button type="submit" className="send-button" disabled={isLoading}>
              {isLoading ? (
                <span className="loading-icon">⏳</span>
              ) : (
                <span className="send-icon">↑</span>
              )}
            </button>
          </div>
        </div>
      </form>
      <div className="output-container">
        <div className="thoughts-window">
          <div className="window-title">思考过程</div>
          <div className="thoughts-content">
            {thoughts.map((thought, index) => (
              <div key={index} className="thought-item">
                {thought}
              </div>
            ))}
          </div>
        </div>
        <div className="result-window">
          <div className="window-title">优化建议</div>
          <div className="response">{response || ''}</div>
        </div>
      </div>
      
      {/* 项目信息（按需加载） */}
      <button 
        className="more-button"
        onClick={() => setShowMore(!showMore)}
      >
        {showMore ? 'Hide details' : 'Show details'}
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