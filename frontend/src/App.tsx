import React, { useState, useCallback, useEffect, useRef } from 'react';
import './App.css';

import { ChatInput, SkillRouteBadge } from './components/chat';
import { Modal } from './components/common';
import { useAbortController, useSubmitControl } from './hooks/useAbortController';
import { useStreamResponse } from './hooks/useStreamResponse';
import { useFileUpload } from './hooks/useFileUpload';
import { useSkillRouter } from './hooks/useSkillRouter';
import { agentService } from './services/agentService';
import Logo from './assets/readyInClient/react.svg';

/** 评分档位 → 配色 */
const RATING_STYLES: Record<string, { color: string; bg: string }> = {
  '优秀': { color: '#16a34a', bg: 'rgba(22, 163, 74, 0.12)' },
  '良好': { color: '#2563eb', bg: 'rgba(37, 99, 235, 0.12)' },
  '一般': { color: '#d97706', bg: 'rgba(217, 119, 6, 0.12)' },
  '较差': { color: '#dc2626', bg: 'rgba(220, 38, 38, 0.12)' },
};

const DIMENSION_LABELS: Record<string, string> = {
  completeness: '完整性',
  professionalism: '专业度',
  quantification: '量化程度',
  matching: '匹配度',
};

function scoreColor(v: number) {
  return v >= 80 ? '#16a34a' : v >= 60 ? '#d97706' : '#dc2626';
}

const PIPELINE_STEPS = [
  { key: 'score', label: '评分', icon: '📊', agent: 'resume_score' },
  { key: 'suggestions', label: '匹配', icon: '🎯', agent: 'jd_match' },
  { key: 'condense', label: '精简', icon: '✂️', agent: 'resume_condense' },
  { key: 'polished', label: '润色', icon: '✨', agent: 'resume_polish' },
] as const;

function App() {
  const [showNewChatModal, setShowNewChatModal] = useState(false);
  const [query, setQuery] = useState('');
  const [jd, setJd] = useState('');
  const [modelProvider, setModelProvider] = useState('deepseek');
  const [copied, setCopied] = useState(false);
  const [exportState, setExportState] = useState<{
    status: 'idle' | 'generating' | 'ready' | 'error';
    url?: string;
    filename?: string;
    size?: number;
    error?: string;
  }>({ status: 'idle' });
  const exportedKeyRef = useRef('');
  const lastUploadedFileRef = useRef<File | null>(null);

  const { create, abort, reset: resetAbort } = useAbortController();
  const { canSubmit, markSubmitting, recordStopTime } = useSubmitControl();

  const {
    score,
    suggestions,
    polished,
    condensed,
    error,
    isStreaming,
    activeAgents,
    elapsed,
    startStream,
    startUpload,
    clearOutput
  } = useStreamResponse();

  const { uploadedFile, handleFileSelect, clearFile } = useFileUpload();

  const { routeState, routeIntent, clearRoute } = useSkillRouter({
    onRoute: () => {},
    onError: (error) => {
      console.error('[SkillRouter] 路由错误:', error);
    }
  });

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    if (e) e.preventDefault();

    if (!canSubmit()) return;
    // 粘贴链路必须有简历文本；上传链路文本作为可选补充要求
    if (!query.trim() && !uploadedFile) return;

    markSubmitting(true);
    clearOutput();
    clearRoute();
    setCopied(false);

    const controller = create();

    try {
      await routeIntent(query, modelProvider);

      if (uploadedFile) {
        // 记录原始文件，供导出时走"原 docx 对齐改写"路径；
        // 上传解析后进入与粘贴链路相同的 MAS 事件流（评分/匹配/精简/润色）
        lastUploadedFileRef.current = uploadedFile;
        await startUpload({
          file: uploadedFile,
          query: query.trim() || undefined,
          jd: jd.trim() || undefined,
          provider: modelProvider
        }, controller.signal);
      } else {
        lastUploadedFileRef.current = null;
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
  }, [query, jd, modelProvider, uploadedFile, canSubmit, markSubmitting, clearOutput, create, startStream, startUpload, clearFile, routeIntent, clearRoute]);

  const handleStop = useCallback(() => {
    recordStopTime();
    abort();
  }, [abort, recordStopTime]);

  const confirmNewChat = useCallback(() => {
    setQuery('');
    setJd('');
    clearOutput();
    clearFile();
    resetAbort();
    lastUploadedFileRef.current = null;
    exportedKeyRef.current = '';
    setExportState(prev => {
      if (prev.url) URL.revokeObjectURL(prev.url);
      return { status: 'idle' };
    });
    setShowNewChatModal(false);
  }, [clearOutput, clearFile, resetAbort]);

  const polishedText = polished?.optimized_resume ?? '';
  const scoreValue = score?.overall_score?.score;
  const rating = score?.overall_score?.rating ?? '';
  const ratingStyle = RATING_STYLES[rating] ?? { color: '#4f5bd5', bg: 'rgba(79, 91, 213, 0.12)' };
  const dimensionAnalysis = (score?.dimension_analysis ?? {}) as Record<
    string,
    { assessment?: string; suggestion?: string } | string
  >;

  const handleCopy = useCallback(async () => {
    if (!polishedText) return;
    try {
      await navigator.clipboard.writeText(polishedText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      console.error('复制失败');
    }
  }, [polishedText]);

  const handleExport = useCallback(async () => {
    if (!polishedText) return;
    setExportState(prev => {
      if (prev.url) URL.revokeObjectURL(prev.url);
      return { status: 'generating' };
    });
    try {
      const { blob, filename } = await agentService.exportDocx(
        polishedText,
        lastUploadedFileRef.current
      );
      const url = URL.createObjectURL(blob);
      setExportState({ status: 'ready', url, filename, size: blob.size });
    } catch (e: unknown) {
      const msg = e instanceof Error
        ? (e.name === 'TimeoutError' ? '文档生成超时（60s），请重试' : e.message)
        : '文档生成失败';
      setExportState({ status: 'error', error: msg });
    }
  }, [polishedText]);

  // 润色完成后自动生成一次优化文档（相同结果不重复生成）
  useEffect(() => {
    if (isStreaming || !polishedText) return;
    const key = `${lastUploadedFileRef.current?.name ?? ''}|${polishedText.length}|${polishedText.slice(-32)}`;
    if (exportedKeyRef.current === key) return;
    exportedKeyRef.current = key;
    handleExport();
  }, [isStreaming, polishedText, handleExport]);

  const handleDownload = useCallback(() => {
    if (!exportState.url) return;
    const a = document.createElement('a');
    a.href = exportState.url;
    a.download = exportState.filename || '简历_优化版.docx';
    a.click();
  }, [exportState]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src={Logo} alt="Logo" className="brand-logo" />
          <div className="brand-text">
            <span className="brand-name">Fulin AI</span>
            <span className="brand-sub">简历优化助手</span>
          </div>
        </div>
        <div className="topbar-actions">
          <div className="model-switch" role="tablist" aria-label="模型选择">
            <button
              type="button"
              className={`model-option${modelProvider === 'deepseek' ? ' active' : ''}`}
              onClick={() => setModelProvider('deepseek')}
            >
              DeepSeek
              <span className="model-tag">在线</span>
            </button>
            <button
              type="button"
              className={`model-option${modelProvider === 'local' ? ' active' : ''}`}
              onClick={() => setModelProvider('local')}
            >
              本地 Qwen
              <span className="model-tag muted">离线</span>
            </button>
          </div>
          <button type="button" className="ghost-button" onClick={() => setShowNewChatModal(true)}>
            新对话
          </button>
        </div>
      </header>

      <main className="workspace">
        {/* 左列：输入 + 优化结果 */}
        <section className="main-col">
          <form className="card input-card" onSubmit={(e) => handleSubmit(e)}>
            <textarea
              className="resume-input"
              placeholder="粘贴简历内容，或点击左下角上传文档 / 图片 / 音频…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={isStreaming}
              rows={7}
            />
            <div className="jd-block">
              <label className="jd-label" htmlFor="jd-input">目标岗位 JD（可选）</label>
              <textarea
                id="jd-input"
                className="jd-input"
                placeholder="粘贴职位描述，匹配分析将围绕它展开"
                value={jd}
                onChange={(e) => setJd(e.target.value)}
                disabled={isStreaming}
                rows={4}
              />
            </div>
            <ChatInput
              onSubmit={() => handleSubmit()}
              onStop={handleStop}
              isLoading={isStreaming}
              uploadedFile={uploadedFile}
              onFileChange={handleFileSelect}
              onFileClear={clearFile}
              disabled={!canSubmit() && isStreaming}
            />
          </form>

          <SkillRouteBadge routeState={routeState} />

          {error && (
            <div className="error-banner" role="alert">
              <span className="error-banner-icon">⚠️</span>
              <span className="error-banner-text">{error}</span>
            </div>
          )}

          {/* 简历评分（维度分析含改进建议） */}
          <div className="card score-card">
            <div className="card-head">
              <span className="card-title">简历评分</span>
            </div>
            <div className="card-body score-body">
              {score && scoreValue != null ? (
                <div className="score-wrap">
                  <div className="score-left">
                    <div
                      className="score-ring"
                      style={{
                        background: `conic-gradient(${ratingStyle.color} ${Math.min(100, Number(scoreValue)) * 3.6}deg, var(--border) 0deg)`
                      }}
                    >
                      <div className="score-ring-inner">
                        <span className="score-number">{scoreValue}</span>
                      </div>
                    </div>
                    <span className="rating-chip" style={{ color: ratingStyle.color, background: ratingStyle.bg }}>
                      {rating || '—'}
                    </span>
                    {score.overall_score?.description && (
                      <p className="score-overall-desc">{score.overall_score.description}</p>
                    )}
                  </div>
                  {score.scores && typeof score.scores === 'object' && (
                    <div className="score-dimensions">
                      {Object.entries(score.scores).map(([key, val]) => {
                        const label = DIMENSION_LABELS[key] || key;
                        const v = Number(val) || 0;
                        const color = scoreColor(v);
                        const item = dimensionAnalysis[key];
                        const assessment = typeof item === 'string' ? item : item?.assessment;
                        const suggestion = typeof item === 'string' ? '' : item?.suggestion;
                        return (
                          <div key={key} className="dim-block">
                            <div className="dim-row">
                              <span className="dim-label">{label}</span>
                              <div className="dim-bar-track">
                                <div className="dim-bar-fill" style={{ width: `${v}%`, background: color }} />
                              </div>
                              <span className="dim-value" style={{ color }}>{v}</span>
                            </div>
                            {assessment && <p className="dim-analysis">{assessment}</p>}
                            {suggestion && (
                              <p className="dim-suggestion">
                                <span className="dim-suggestion-icon">💡</span>
                                {suggestion}
                              </p>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              ) : (
                <div className="empty-state compact">
                  <span className="empty-icon">📊</span>
                  <p className="empty-sub">发送简历后出分</p>
                </div>
              )}
            </div>
          </div>

          <div className="card result-card">
            <div className="card-head">
              <span className="card-title">优化结果</span>
              <div className="card-head-actions">
                {isStreaming && <span className="streaming-hint">生成中<span className="dot-pulse" /></span>}
                {polishedText && (
                  <button type="button" className="ghost-button small" onClick={handleCopy}>
                    {copied ? '已复制 ✓' : '复制全文'}
                  </button>
                )}
              </div>
            </div>
            {(isStreaming || score || suggestions || polished) && (
              <div className="pipeline-bar">
                {PIPELINE_STEPS
                  // 精简为条件步骤：仅在触发（活跃或已完成）时显示
                  .filter(step => step.key !== 'condense' || !!condensed || activeAgents.has(step.agent))
                  .map((step, i, arr) => {
                    const arrived = step.key === 'score' ? !!score
                      : step.key === 'suggestions' ? !!suggestions
                      : step.key === 'condense' ? !!condensed
                      : !!polished;
                    const isActive = isStreaming && !arrived && activeAgents.has(step.agent);
                    const isCurrent = isStreaming && !arrived && !isActive && (
                      i === 0 ? true : !!({
                        score: !!score,
                        suggestions: !!suggestions,
                        condense: !!condensed,
                        polished: !!polished,
                      }[arr[i - 1].key])
                    );
                    return (
                      <React.Fragment key={step.key}>
                        <div className={`pipeline-step${arrived ? ' done' : ''}${(isActive || isCurrent) ? ' active' : ''}`}>
                          <span className="step-icon">{arrived ? '✓' : step.icon}</span>
                          <span className="step-label">{step.label}</span>
                        </div>
                        {i < arr.length - 1 && <div className="pipeline-connector" />}
                      </React.Fragment>
                    );
                  })}
                {elapsed != null && !isStreaming && (
                  <span className="pipeline-elapsed">{elapsed}s</span>
                )}
              </div>
            )}
            <div className="card-body result-body">
              {polishedText ? (
                <pre className="result-text">{polishedText}</pre>
              ) : (
                <div className="empty-state">
                  <span className="empty-icon">✨</span>
                  <p className="empty-title">优化后的简历将在这里流式输出</p>
                  <p className="empty-sub">在上方输入简历内容，选择模型后点击发送</p>
                </div>
              )}
            </div>
          </div>

          {exportState.status !== 'idle' && (
            <div className="card export-card">
              <div className="card-head">
                <span className="card-title">优化文档</span>
                {exportState.status === 'ready' && (
                  <button type="button" className="export-regen" onClick={handleExport}>
                    重新生成
                  </button>
                )}
              </div>
              <div className="card-body">
                {exportState.status === 'generating' && (
                  <div className="export-generating">
                    正在生成优化文档<span className="dot-pulse" />
                  </div>
                )}
                {exportState.status === 'ready' && (
                  <div className="doc-entry">
                    <span className="doc-icon">📄</span>
                    <div className="doc-info">
                      <span className="doc-name">{exportState.filename}</span>
                      <span className="doc-size">{Math.max(1, Math.round((exportState.size ?? 0) / 1024))} KB</span>
                    </div>
                    <button type="button" className="export-download" onClick={handleDownload}>
                      下载 .docx
                    </button>
                  </div>
                )}
                {exportState.status === 'error' && (
                  <div className="export-error">
                    <span>文档生成失败：{exportState.error}</span>
                    <button type="button" className="export-retry" onClick={handleExport}>重试</button>
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
      </main>

      <Modal
        isOpen={showNewChatModal}
        onClose={() => setShowNewChatModal(false)}
        onConfirm={confirmNewChat}
        title="开启新对话"
        message="这将清空当前输入与所有结果"
        confirmText="确认"
        cancelText="取消"
      />
    </div>
  );
}

export default App;
