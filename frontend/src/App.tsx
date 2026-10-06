import React, { useState, Suspense, useCallback } from 'react';
import './App.css';

// 导入新架构组件和 Hooks
import { ChatInput, SkillRouteBadge } from './components/chat';
import { Modal } from './components/common';
import { useAbortController, useSubmitControl } from './hooks/useAbortController';
import { useStreamResponse } from './hooks/useStreamResponse';
import { useFileUpload } from './hooks/useFileUpload';
import { useSkillRouter } from './hooks/useSkillRouter';
import Logo from './assets/readyInClient/react.svg';

const NonCriticalComponent = React.lazy(() => 
  import('./components/NonCriticalComponent')
);

function App() {
  const [showMore, setShowMore] = useState(false);
  const [showNewChatModal, setShowNewChatModal] = useState(false);
  const [query, setQuery] = useState('');
  const [jd, setJd] = useState(''); // JD 输入
  const [modelProvider, setModelProvider] = useState('local'); // 模型切换
  
  const { create, abort, reset: resetAbort } = useAbortController();
  const { canSubmit, markSubmitting, recordStopTime } = useSubmitControl();
  
  // 使用新的 useStreamResponse - 返回 score, suggestions, polished
  const {
    score,
    suggestions,
    polished,
    isStreaming,
    startStream,
    applyPolished,
    clearOutput
  } = useStreamResponse();

  const { uploadedFile, handleFileSelect, uploadFile, clearFile } = useFileUpload();
  
  // Skill Router - 意图识别
  const { routeState, routeIntent, clearRoute } = useSkillRouter({
    onRoute: (route) => {
      console.log(`[SkillRouter] 识别到技能: ${route.skill} (来源: ${route.source}, 置信度: ${route.confidence})`);
    },
    onError: (error) => {
      console.error('[SkillRouter] 路由错误:', error);
    }
  });

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    
    if (!canSubmit()) return;
    if (!query.trim()) return;
    
    markSubmitting(true);
    clearOutput();
    clearRoute();
    
    const controller = create();
    
    try {
      // 先进行意图识别
      const route = await routeIntent(query);
      console.log('[SkillRouter] 路由结果:', route);
      
      if (uploadedFile) {
        // 文件上传：multipart 提交 → 后端解析 PDF/DOCX → 流式优化（结果上屏到润色区）
        let acc = '';
        await uploadFile(query, controller.signal, (token) => {
          acc += token;
        });
        applyPolished(acc);
      } else {
        // 直接优化简历
        await startStream({
          resume: query,
          jd: jd || undefined,
          provider: modelProvider
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
  }, [query, jd, modelProvider, uploadedFile, canSubmit, markSubmitting, clearOutput, create, startStream, applyPolished, uploadFile, clearFile, routeIntent, clearRoute]);

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
        <div className="model-switcher">
          <label htmlFor="model-provider" className="model-switcher-label">模型:</label>
          <select
            id="model-provider"
            className="model-switcher-select"
            value={modelProvider}
            onChange={(e) => setModelProvider(e.target.value)}
          >
            <option value="local">本地 Qwen</option>
            <option value="deepseek">DeepSeek</option>
          </select>
        </div>
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
              onFileClear={clearFile}
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

      {/* Skill Router 意图识别状态 */}
      <div className="skill-route-container">
        <SkillRouteBadge routeState={routeState} />
      </div>

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
