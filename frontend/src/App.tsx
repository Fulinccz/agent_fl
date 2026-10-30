import React, { useEffect, useRef, useState } from 'react';
import './App.css';

import { Modal } from './components/common';
import { useSubmitControl } from './hooks/useAbortController';
import { useStreamResponse, type UseStreamResponseReturn } from './hooks/useStreamResponse';
import { ProfileProvider } from './pages/ProfileProvider';
import { useProfile } from './pages/profile-context';
import ProfileForm from './pages/ProfileForm';
import AnalyzePage from './pages/AnalyzePage';
import GeneratePage from './pages/GeneratePage';
import InterviewPage from './pages/InterviewPage';
import type { PageKey } from './types';
import Logo from './assets/readyInClient/react.svg';

/** 顶部三页入口：hash 路由，不引入 react-router */
const PAGE_TABS: { key: PageKey; label: string; icon: string }[] = [
  { key: 'analyze', label: '简历分析', icon: '📊' },
  { key: 'generate', label: '简历生成', icon: '✨' },
  { key: 'interview', label: '面试攻略', icon: '💼' },
];

const PAGE_KEYS = PAGE_TABS.map((tab) => tab.key);

function readPageFromHash(): PageKey {
  const key = window.location.hash.replace(/^#\/?/, '');
  return (PAGE_KEYS as string[]).includes(key) ? (key as PageKey) : 'analyze';
}

/**
 * 三页工作台
 *
 * - 输入（简历 / JD / 附件 / 语音）来自 ProfileContext，三页共用一份
 * - 结果按页分桶：每页一个 useStreamResponse，页面常驻挂载（hidden 切换），
 *   切页既不断流也不互相清空
 * - 全局同一时刻只允许一条推理流，避免本地 llama-server 队列堵塞
 */
function Workspace() {
  const profile = useProfile();
  const [page, setPage] = useState<PageKey>(readPageFromHash);
  const [showNewChatModal, setShowNewChatModal] = useState(false);
  // 新对话时递增：作为 GeneratePage 的 key，重挂载清空导出状态与 blob URL
  const [resetToken, setResetToken] = useState(0);
  // 生成页的原文对照：记录提交那一刻的简历文本与原始文件
  const [generateSource, setGenerateSource] = useState<{ text: string; file: File | null }>({
    text: '',
    file: null
  });
  // 分析页的匹配卡只在「提交时确实带了 JD」的这一轮出现
  const [analyzeJd, setAnalyzeJd] = useState('');
  const controllerRef = useRef<AbortController | null>(null);
  // 生成页上传提交时，等 resume_parsed 回来后补全原文对照
  const pendingGenerateTextRef = useRef(false);
  const { canSubmit, markSubmitting, recordStopTime } = useSubmitControl();

  const handleResumeParsed = (text: string) => {
    profile.backfillResume(text);
    if (pendingGenerateTextRef.current) {
      pendingGenerateTextRef.current = false;
      setGenerateSource((prev) => ({ ...prev, text }));
    }
  };

  const analyzeStream = useStreamResponse({ onResumeParsed: handleResumeParsed });
  const generateStream = useStreamResponse({ onResumeParsed: handleResumeParsed });
  const interviewStream = useStreamResponse({ onResumeParsed: handleResumeParsed });

  const streams: Record<PageKey, UseStreamResponseReturn> = {
    analyze: analyzeStream,
    generate: generateStream,
    interview: interviewStream
  };

  useEffect(() => {
    const sync = () => setPage(readPageFromHash());
    window.addEventListener('hashchange', sync);
    if (!window.location.hash) window.location.replace('#/analyze');
    return () => window.removeEventListener('hashchange', sync);
  }, []);

  const navigate = (key: PageKey) => {
    if (window.location.hash === `#/${key}`) {
      setPage(key);
      return;
    }
    window.location.hash = `#/${key}`;
  };

  const streamingPage = PAGE_KEYS.find((key) => streams[key].isStreaming) ?? null;
  const hasInput = profile.query.trim() !== '' || !!profile.uploadedFile;

  const submit = async (target: PageKey) => {
    if (streamingPage !== null) return;
    if (!canSubmit()) return;
    // 录音中/转写中不提交，避免把未落定的语音内容送进流水线
    if (profile.voiceState !== 'idle') return;
    // 粘贴链路必须有简历文本；上传链路文本作为可选补充要求
    if (!hasInput) return;

    markSubmitting(true);
    const controller = new AbortController();
    controllerRef.current = controller;
    const stream = streams[target];

    if (target === 'generate') {
      // 上传链路先记文件，原文对照等 resume_parsed 回来再补文本
      pendingGenerateTextRef.current = !!profile.uploadedFile;
      setGenerateSource({
        text: profile.uploadedFile ? '' : profile.query,
        file: profile.uploadedFile
      });
    }
    if (target === 'analyze') {
      setAnalyzeJd(profile.jd);
    }

    try {
      if (profile.uploadedFile) {
        // 上传解析后进入与粘贴链路相同的 MAS 事件流；文件保留在三页输入中
        await stream.startUpload({
          file: profile.uploadedFile,
          // 自动回填的简历正文不作为"补充要求"，避免润色提示词里重复整份简历
          query: profile.queryAutoFilled ? undefined : profile.query.trim() || undefined,
          jd: profile.jd.trim() || undefined,
          provider: profile.provider,
          page: target
        }, controller.signal);
      } else {
        await stream.startStream({
          resume: profile.query,
          jd: profile.jd.trim() || undefined,
          provider: profile.provider,
          page: target
        }, controller.signal);
      }
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
      markSubmitting(false);
    }
  };

  const stop = () => {
    recordStopTime();
    controllerRef.current?.abort();
  };

  const confirmNewChat = () => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    analyzeStream.clearOutput();
    generateStream.clearOutput();
    interviewStream.clearOutput();
    profile.resetInputs();
    setGenerateSource({ text: '', file: null });
    setAnalyzeJd('');
    pendingGenerateTextRef.current = false;
    setResetToken((token) => token + 1);
    setShowNewChatModal(false);
  };

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

        <nav className="page-tabs" role="tablist" aria-label="功能页面">
          {PAGE_TABS.map((tab) => (
            <button
              key={tab.key}
              type="button"
              role="tab"
              aria-selected={page === tab.key}
              className={`page-tab${page === tab.key ? ' active' : ''}`}
              onClick={() => navigate(tab.key)}
            >
              <span className="page-tab-icon">{tab.icon}</span>
              {tab.label}
              {streams[tab.key].isStreaming && <span className="page-tab-dot" />}
            </button>
          ))}
        </nav>

        <div className="topbar-actions">
          <div className="model-switch" role="tablist" aria-label="模型选择">
            <button
              type="button"
              className={`model-option${profile.provider === 'deepseek' ? ' active' : ''}`}
              onClick={() => profile.setProvider('deepseek')}
            >
              DeepSeek
              <span className="model-tag">在线</span>
            </button>
            <button
              type="button"
              className={`model-option${profile.provider === 'local' ? ' active' : ''}`}
              onClick={() => profile.setProvider('local')}
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
        {PAGE_KEYS.map((key) => {
          const stream = streams[key];
          const isCurrent = key === page;
          return (
            <section
              key={key}
              className="page-panel"
              role="tabpanel"
              hidden={!isCurrent}
              aria-hidden={!isCurrent}
            >
              <ProfileForm
                page={key}
                isStreaming={stream.isStreaming}
                blockedByOther={streamingPage !== null && streamingPage !== key}
                onSubmit={() => { void submit(key); }}
                onStop={stop}
              />

              {isCurrent && profile.voiceError && (
                <div className="error-banner" role="alert">
                  <span className="error-banner-icon">🎤</span>
                  <span className="error-banner-text">{profile.voiceError}</span>
                  <button
                    type="button"
                    className="error-banner-close"
                    onClick={() => profile.setVoiceError('')}
                    title="关闭"
                  >
                    ✕
                  </button>
                </div>
              )}

              {isCurrent && stream.error && (
                <div className="error-banner" role="alert">
                  <span className="error-banner-icon">⚠️</span>
                  <span className="error-banner-text">{stream.error}</span>
                  <button
                    type="button"
                    className="error-banner-close"
                    onClick={stream.clearError}
                    title="关闭"
                  >
                    ✕
                  </button>
                </div>
              )}

              {key === 'analyze' && <AnalyzePage stream={stream} hasJd={analyzeJd.trim() !== ''} />}

              {key === 'generate' && (
                <GeneratePage
                  key={resetToken}
                  stream={stream}
                  sourceText={generateSource.text}
                  sourceFile={generateSource.file}
                />
              )}

              {key === 'interview' && <InterviewPage stream={stream} />}
            </section>
          );
        })}
      </main>

      <Modal
        isOpen={showNewChatModal}
        onClose={() => setShowNewChatModal(false)}
        onConfirm={confirmNewChat}
        title="开启新对话"
        message="这将清空三个页面的输入与结果"
        confirmText="确认"
        cancelText="取消"
      />
    </div>
  );
}

function App() {
  return (
    <ProfileProvider>
      <Workspace />
    </ProfileProvider>
  );
}

export default App;
