import React, { useState, Suspense, useCallback } from 'react';
import './App.css';

// 导入新架构组件和 Hooks
import { ChatInput, ChatOutput } from './components/chat';
import { Modal } from './components/common';
import { useAbortController, useSubmitControl } from './hooks/useAbortController';
import { useStreamResponse } from './hooks/useStreamResponse';
import { useFileUpload } from './hooks/useFileUpload';
import Logo from './assets/readyInClient/react.svg';

const NonCriticalComponent = React.lazy(() => 
  import('./components/NonCriticalComponent')
);

function App() {
  const [showMore, setShowMore] = useState(false);
  const [showNewChatModal, setShowNewChatModal] = useState(false);
  const [query, setQuery] = useState('');
  const [jd, setJd] = useState(''); // JD 输入
  
  const { create, abort, reset: resetAbort } = useAbortController();
  const { canSubmit, markSubmitting, recordStopTime } = useSubmitControl();
  
  // 使用新的 useStreamResponse - 返回 score, suggestions, polished
  const { 
    score, 
    suggestions, 
    polished, 
    isStreaming, 
    startStream, 
    clearOutput 
  } = useStreamResponse();
  
  const { uploadedFile, handleFileSelect, clearFile } = useFileUpload();

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    
    if (!canSubmit()) return;
    if (!query.trim()) return;
    
    markSubmitting(true);
    clearOutput();
    
    const controller = create();
    
    try {
      if (uploadedFile) {
        // 文件上传处理 - 读取文件内容
        const fileContent = await uploadedFile.text();
        await startStream({
          resume: fileContent,
          jd: jd || undefined
        }, controller.signal);
      } else {
        // 直接优化简历
        await startStream({
          resume: query,
          jd: jd || undefined
        }, controller.signal);
      }
    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') {
        console.error('Error:', error);
      }
    } finally {
      markSubmitting(false);
      clearFile();
    }
  }, [query, jd, uploadedFile, canSubmit, markSubmitting, clearOutput, create, startStream, clearFile]);

  const handleStop = useCallback(() => {
    recordStopTime();
    abort();
  }, [abort, recordStopTime]);

  const handleNewChat = useCallback(() => {
    setShowNewChatModal(true);
  }, []);

  const confirmNewChat = useCallback(() => {
    setQuery('');
    setJd('');
    clearOutput();
    clearFile();
    resetAbort();
    setShowNewChatModal(false);
  }, [clearOutput, clearFile, resetAbort]);

  // 格式化评分显示
  const formatScore = () => {
    if (!score) return '';
    const { overall_score, scores } = score;
    return `综合评分: ${overall_score?.score}分 (${overall_score?.rating})

各维度评分:
- 完整性: ${scores?.completeness}分
- 专业度: ${scores?.professionalism}分  
- 量化程度: ${scores?.quantification}分
- 匹配度: ${scores?.matching}分`;
  };

  // 格式化建议显示
  const formatSuggestions = () => {
    if (!suggestions?.suggestions) return '';
    return suggestions.suggestions.map((s: any, i: number) => 
      `${i + 1}. [${s.category}] ${s.suggestion}${s.example ? '\n   示例: ' + s.example : ''}`
    ).join('\n\n');
  };

  // 格式化润色结果显示
  const formatPolished = () => {
    if (!polished?.optimized_resume) return '';
    return polished.optimized_resume;
  };

  return (
    <div className="app">
      <div className="header">
        <img src={Logo} alt="Logo" className="logo" />
        <h1>Fulin AI</h1>
      </div>
      
      <form onSubmit={(e) => handleSubmit(e)} className="form">
        <div className="input-row">
          {/* 简历输入框 */}
          <div className="input-section">
            <ChatInput
              value={query}
              onChange={setQuery}
              onSubmit={() => handleSubmit()}
              onStop={handleStop}
              isLoading={isStreaming}
              uploadedFile={uploadedFile}
              onFileChange={handleFileSelect}
              onNewChat={handleNewChat}
              disabled={!canSubmit() && isStreaming}
            />
          </div>
          
          {/* JD 输入框 */}
          <div className="input-section jd-section">
            <div className="jd-input-wrapper">
              <textarea
                className="jd-input"
                placeholder="目标岗位 JD（可选）"
                value={jd}
                onChange={(e) => setJd(e.target.value)}
              />
            </div>
          </div>
        </div>
      </form>

      <div className="output-container">
        {/* 分区1：简历评分 - 紧凑单行显示 */}
        <div className="output-section score-section">
          <div className="score-header">
            <div className="score-title-group">
              <span className="section-icon">📊</span>
              <span className="section-title">简历评分</span>
            </div>
            {score ? (
              <div className="score-display">
                <span className="score-number">{score.overall_score?.score}</span>
                <span className="score-rating">{score.overall_score?.rating}</span>
              </div>
            ) : (
              <span className="score-placeholder">等待评分...</span>
            )}
          </div>
        </div>
        
        {/* 分区2：优化建议 - 中等高度 */}
        <div className="output-section suggestions-section">
          <div className="section-header">
            <span className="section-icon">💡</span>
            <span className="section-title">优化建议</span>
          </div>
          <div className="section-content">
            {suggestions ? (
              <pre className="output-text suggestions-text">{formatSuggestions()}</pre>
            ) : (
              <div className="section-placeholder">等待优化建议...</div>
            )}
          </div>
        </div>
        
        {/* 分区3：优化结果 - 最大高度 */}
        <div className="output-section polished-section">
          <div className="section-header">
            <span className="section-icon">✨</span>
            <span className="section-title">优化结果</span>
          </div>
          <div className="section-content">
            {polished ? (
              <pre className="output-text">{formatPolished()}</pre>
            ) : (
              <div className="section-placeholder">等待优化后的简历...</div>
            )}
          </div>
        </div>
      </div>

      <button 
        className="more-button"
        onClick={() => setShowMore(!showMore)}
      >
        {showMore ? 'Hide details' : 'Show details'}
      </button>

      {showMore && (
        <div className="details-container">
          <Suspense fallback={<div className="loading">Loading...</div>}>
            <NonCriticalComponent />
          </Suspense>
        </div>
      )}

      <Modal
        isOpen={showNewChatModal}
        onClose={() => setShowNewChatModal(false)}
        onConfirm={confirmNewChat}
        title="开启新对话"
        message="这将清空当前消息记录"
        confirmText="确认"
        cancelText="取消"
      />
    </div>
  );
}

export default App;
