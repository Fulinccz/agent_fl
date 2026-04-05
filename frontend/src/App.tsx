import React, { useState, Suspense, useCallback } from 'react';
import './App.css';

// 导入新架构组件和Hooks
import { ChatInput, ChatOutput, ThoughtProcess } from './components/chat';
import { Modal } from './components/common';
import { useAbortController, useSubmitControl } from './hooks/useAbortController';
import { useStreamResponse } from './hooks/useStreamResponse';
import { useFileUpload } from './hooks/useFileUpload';
import { agentService } from './services/agentService';
import Logo from './assets/readyInClient/react.svg';

const NonCriticalComponent = React.lazy(() => 
  import('./components/NonCriticalComponent')
);

function App() {
  const [showMore, setShowMore] = useState(false);
  const [showNewChatModal, setShowNewChatModal] = useState(false);
  const [query, setQuery] = useState('');
  const [deepThinking, setDeepThinking] = useState(false);
  
  const { abortController, create, abort, reset: resetAbort } = useAbortController();
  const { canSubmit, markSubmitting, recordStopTime } = useSubmitControl();
  const { thoughts, response, isStreaming, startStream, clearOutput } = useStreamResponse();
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
        await agentService.uploadAndProcess(
          { file: uploadedFile, query },
          controller.signal
        );
      } else {
        await startStream(query, controller.signal, deepThinking);
      }
    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') {
        console.error('Error:', error);
      }
    } finally {
      markSubmitting(false);
      clearFile();
    }
  }, [query, uploadedFile, canSubmit, markSubmitting, clearOutput, create, startStream, clearFile]);

  const handleStop = useCallback(() => {
    recordStopTime();
    abort();
  }, [abort, recordStopTime]);

  const handleNewChat = useCallback(() => {
    setShowNewChatModal(true);
  }, []);

  const handleDeepThinkToggle = useCallback(() => {
    setDeepThinking(prev => !prev);
  }, []);

  const confirmNewChat = useCallback(() => {
    setQuery('');
    clearOutput();
    clearFile();
    resetAbort();
    setShowNewChatModal(false);
  }, [clearOutput, clearFile, resetAbort]);

  return (
    <div className="app">
      <div className="header">
        <img src={Logo} alt="Logo" className="logo" />
        <h1>Fulin AI</h1>
      </div>
      
      <form onSubmit={(e) => handleSubmit(e)} className="form">
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
          deepThinking={deepThinking}
          onDeepThinkToggle={handleDeepThinkToggle}
        />
      </form>

      <div className="output-container">
        <ThoughtProcess 
          thoughts={thoughts} 
          isStopped={abortController?.signal.aborted || false}
        />
        
        <ChatOutput
          content={response}
          title="优化建议"
          className="result-window"
        />
      </div>

      <button 
        className="more-button"
        onClick={() => setShowMore(!showMore)}
      >
        {showMore ? 'Hide details' : 'Show details'}
      </button>

      {showMore && (
        <Suspense fallback={<div className="loading">Loading...</div>}>
          <NonCriticalComponent />
        </Suspense>
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
