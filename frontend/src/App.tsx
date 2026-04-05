// 核心组件直接导入（本地加载）
import React, { useState, Suspense, useRef } from 'react';
import './App.css';
// 关键资源直接导入（修正路径）
import Logo from './assets/readyInClient/react.svg';

// 使用React.lazy动态导入非关键组件
const NonCriticalComponent = React.lazy(() => 
  import('./components/NonCriticalComponent')
);

function App() {
  // 状态管理
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [thoughts, setThoughts] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showMore, setShowMore] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [showNewChatModal, setShowNewChatModal] = useState(false);
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  
  // 使用 ref 来跟踪是否已经停止
  const isStoppedRef = useRef(false);
  
  // 使用 ref 来跟踪是否正在提交中（防止重复提交）
  const isSubmittingRef = useRef(false);
  
  // 使用 ref 来跟踪上次停止的时间戳（防止停止后立即重试）
  const lastStopTimeRef = useRef(0);
  
  // 冷却期：停止后2秒内不允许重新提交
  const STOP_COOLDOWN_MS = 2000;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setUploadedFile(e.target.files[0]);
    }
  };

  // 常量定义
  const baseUrl = (import.meta.env.VITE_API_URL as string) || '/api';

  // 处理文件上传
  const handleFileUpload = async (controller: AbortController) => {
    // 检查是否已经中止
    if (controller.signal.aborted) {
      console.log('Upload already aborted');
      return;
    }
    
    try {
      const formData = new FormData();
      formData.append('file', uploadedFile!);
      formData.append('query', query);
      formData.append('provider', 'local');
      
      const res = await fetch(`${baseUrl}/agent/upload`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });
      
      if (!res.ok) throw new Error('File upload failed');
      
      const data = await res.json();
      setResponse(data.response);
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        // 中止错误，直接返回
        console.log('Upload aborted');
        return;
      } else {
        throw error;
      }
    }
  };

  // 处理流式响应
  const handleStreamResponse = async (controller: AbortController) => {
    console.log('=== handleStreamResponse 开始 ===');
    console.log('signal.aborted:', controller.signal.aborted);
    
    // 检查是否已经中止
    if (controller.signal.aborted) {
      console.log('⚠️ Stream already aborted, 直接返回');
      return;
    }
    
    try {
      console.log('📤 发送 fetch 请求到:', `${baseUrl}/agent/stream`);
      const res = await fetch(`${baseUrl}/agent/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, model: 'model_serving/Qwen3___5-0___8B' }),
        signal: controller.signal
      });
      
      console.log('📥 收到响应, status:', res.status);
      
      if (!res.body) throw new Error('No response body');
      
      const reader = res.body.getReader();
      let buffer = '';
      let lineCount = 0;
      
      while (true) {
        // 检查是否中止
        if (controller.signal.aborted) {
          console.log(`⏹️ Stream aborted (已处理 ${lineCount} 行), 取消读取器`);
          reader.cancel();
          return;
        }
        
        // 读取数据
        const result = await reader.read();
        
        // 检查是否读取完成
        if (result.done) {
          console.log(`✅ Stream 完成 (共 ${lineCount} 行)`);
          break;
        }
        
        // 解码并处理数据
        buffer += new TextDecoder().decode(result.value);
        
        // 分割可能的多个JSON对象
        const lines = buffer.split('\n');
        for (let i = 0; i < lines.length - 1; i++) {
          const line = lines[i].trim();
          if (line) {
            lineCount++;
            try {
              const data = JSON.parse(line);
              if (data.type === 'thought') {
                setThoughts(prev => [...prev, data.content]);
              } else if (data.type === 'token' && data.full_text) {
                setResponse(data.full_text);
              } else if (data.type === 'complete' && data.full_text) {
                setResponse(data.full_text);
              } else if (data.type === 'error') {
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
      console.log('=== handleStreamResponse 正常结束 ===');
    } catch (error) {
      console.log('❌ handleStreamResponse 捕获到错误:', error instanceof Error ? error.name : error);
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('⏹️ Fetch aborted (AbortError)');
        return;
      } else {
        throw error;
      }
    }
  };

  // 提交表单
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    console.log('=== handleSubmit 开始 ===');
    console.log('当前 isLoading:', isLoading);
    console.log('当前 isStoppedRef:', isStoppedRef.current);
    console.log('当前 isSubmittingRef:', isSubmittingRef.current);
    console.log('距上次停止时间:', Date.now() - lastStopTimeRef.current, 'ms');
    
    // 检查是否已经在加载中，如果是则不重复提交
    if (isLoading) {
      console.log('⚠️ 已在加载中（isLoading），跳过提交');
      return;
    }
    
    // 检查是否正在提交中（使用 ref 防止重复提交）
    if (isSubmittingRef.current) {
      console.log('⚠️ 正在提交中（isSubmittingRef），跳过提交');
      return;
    }
    
    // 检查是否在冷却期内（防止停止后立即重试）
    const timeSinceLastStop = Date.now() - lastStopTimeRef.current;
    if (timeSinceLastStop < STOP_COOLDOWN_MS) {
      console.log(`⚠️ 在冷却期内（${timeSinceLastStop}ms < ${STOP_COOLDOWN_MS}ms），跳过提交`);
      return;
    }
    
    // 标记为正在提交
    isSubmittingRef.current = true;
    console.log('✅ 设置 isSubmittingRef = true');
    
    // 重置停止标志
    isStoppedRef.current = false;
    console.log('✅ 重置 isStoppedRef 为 false');
    
    // 清空之前的输出
    setIsLoading(true);
    setResponse('');
    setThoughts([]);
    console.log('✅ 清空输出，设置 isLoading=true');
    
    // 创建新的 AbortController
    const controller = new AbortController();
    setAbortController(controller);
    console.log('✅ 创建新的 AbortController');
    
    try {
      if (uploadedFile) {
        console.log('📤 开始文件上传...');
        await handleFileUpload(controller);
        console.log('📤 文件上传完成');
      } else {
        console.log('📤 开始流式响应...');
        await handleStreamResponse(controller);
        console.log('📤 流式响应完成');
      }
    } catch (error) {
      console.error('❌ handleSubmit 捕获到错误:', error);
      console.log('当前 isStoppedRef:', isStoppedRef.current);
      
      // 只有在没有停止的情况下才处理错误
      if (!isStoppedRef.current) {
        if (error instanceof Error && (error.name === 'AbortError' || error.message === 'AbortError')) {
          console.log('⏹️ 用户中止生成（错误处理）');
        } else {
          const errorMessage = 'Error: ' + (error instanceof Error ? error.message : String(error));
          setResponse(errorMessage);
          setThoughts(prev => [...prev, `[错误] ${error instanceof Error ? error.message : String(error)}`]);
        }
      } else {
        console.log('⏹️ 用户已停止，忽略错误');
      }
    } finally {
      console.log('=== finally 块 ===');
      console.log('当前 isStoppedRef:', isStoppedRef.current);
      
      // 只有在没有停止的情况下才更新状态
      if (!isStoppedRef.current) {
        console.log('✅ 正常完成，更新状态');
        setIsLoading(false);
        setAbortController(null);
        setUploadedFile(null);
      } else {
        console.log('⏹️ 用户已停止，不在 finally 中更新状态');
      }
      
      // 无论成功、失败还是停止，都重置提交标志
      isSubmittingRef.current = false;
      console.log('✅ 重置 isSubmittingRef = false');
      console.log('=== handleSubmit 结束 ===');
    }
  };

  // 停止生成
  const handleStop = () => {
    console.log('=== handleStop 被调用 ===');
    console.log('当前 abortController:', abortController);
    console.log('当前 isLoading:', isLoading);
    
    if (abortController) {
      console.log('✅ 开始停止流程...');
      
      // 记录停止时间（用于冷却期机制）
      lastStopTimeRef.current = Date.now();
      console.log('✅ 记录停止时间');
      
      // 设置停止标志
      isStoppedRef.current = true;
      console.log('✅ 设置 isStoppedRef = true');
      
      // 中止请求
      abortController.abort();
      console.log('✅ 调用 abortController.abort()');
      console.log('  signal.aborted:', abortController.signal.aborted);
      
      // 立即更新状态，确保停止按钮变回发送按钮
      setIsLoading(false);
      setAbortController(null);
      console.log('✅ 更新状态: isLoading=false, abortController=null');
      
      // 添加停止提示，保留已生成的思考过程
      setThoughts(prev => {
        console.log('📝 添加停止提示, 当前 thoughts 数量:', prev.length);
        return [...prev, '用户已中止生成'];
      });
      
      console.log('=== handleStop 完成 ===');
    } else {
      console.log('⚠️ 没有 abortController，无法停止');
    }
  };

  // 开启新对话
  const handleNewChat = () => {
    setShowNewChatModal(true);
  };

  // 确认新对话
  const confirmNewChat = () => {
    setQuery('');
    setResponse('');
    setThoughts([]);
    setUploadedFile(null);
    setShowNewChatModal(false);
  };

  // 取消新对话
  const cancelNewChat = () => {
    setShowNewChatModal(false);
  };

  return (
    <div className="app">
      <div className="header">
        <img src={Logo} alt="Logo" className="logo" />
        <h1>Fulin AI</h1>

      </div>
      

      
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
          </div>
          <div className="form-actions">
            <div className="left-actions">
              <button 
                type="button" 
                className="new-chat-button"
                onClick={handleNewChat}
              >
                开启新对话
              </button>
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
            <button 
              type={isLoading ? "button" : "submit"} 
              className="send-button" 
              disabled={!isLoading && !query.trim()}
              onClick={isLoading ? handleStop : undefined}
            >
              {isLoading ? (
                <span className="stop-icon">⏹</span>
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
      
      {/* 新对话弹窗 */}
      {showNewChatModal && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>开启新对话</h3>
            <p>这将清空当前消息记录</p>
            <div className="modal-actions">
              <button className="modal-cancel" onClick={cancelNewChat}>
                取消
              </button>
              <button className="modal-confirm" onClick={confirmNewChat}>
                确认
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;