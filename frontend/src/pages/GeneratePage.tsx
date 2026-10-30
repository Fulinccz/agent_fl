import React, { useCallback, useEffect, useRef, useState } from 'react';
import type { UseStreamResponseReturn } from '../hooks/useStreamResponse';
import { agentService } from '../services/agentService';

interface GeneratePageProps {
  stream: UseStreamResponseReturn;
  /** 提交时的简历原文（用于原文对照） */
  sourceText: string;
  /** 提交时的原始上传文件（docx 导出走对齐改写路径） */
  sourceFile: File | null;
}

type ExportState = {
  status: 'idle' | 'generating' | 'ready' | 'error';
  url?: string;
  previewUrl?: string;
  filename?: string;
  size?: number;
  error?: string;
};

/** 简历生成页：成品主视图 + 可折叠原文对照 + 复制 / 导出 docx / 只读预览 */
const GeneratePage: React.FC<GeneratePageProps> = ({ stream, sourceText, sourceFile }) => {
  const { polished, condensed, isStreaming, elapsed, liveSeconds, thoughts } = stream;
  const polishedText = polished?.optimized_resume ?? '';

  const [copied, setCopied] = useState(false);
  const [copiedPath, setCopiedPath] = useState(false);
  const [exportState, setExportState] = useState<ExportState>({ status: 'idle' });
  const exportedKeyRef = useRef('');

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
    setExportState({ status: 'generating' });
    try {
      const result = await agentService.exportDocx(polishedText, sourceFile);
      setExportState({
        status: 'ready',
        url: result.url,
        previewUrl: result.preview_url,
        filename: result.filename,
        size: result.size
      });
    } catch (e: unknown) {
      const msg = e instanceof Error
        ? (e.name === 'TimeoutError' ? '文档生成超时（60s），请重试' : e.message)
        : '文档生成失败';
      setExportState({ status: 'error', error: msg });
    }
  }, [polishedText, sourceFile]);

  // 润色完成后自动生成一次优化文档（相同结果不重复生成）
  useEffect(() => {
    if (isStreaming || !polishedText) return;
    const key = `${sourceFile?.name ?? ''}|${polishedText.length}|${polishedText.slice(-32)}`;
    if (exportedKeyRef.current === key) return;
    exportedKeyRef.current = key;
    // 放到下一轮事件循环：effect 内同步 setState 会触发级联渲染
    const timer = window.setTimeout(() => { void handleExport(); }, 0);
    return () => window.clearTimeout(timer);
  }, [isStreaming, polishedText, sourceFile, handleExport]);

  const fileUrl = exportState.url ?? '';
  // 展示可点击的完整地址（相对路径经 Vite/网关代理同样可用）
  const fullPath = fileUrl ? `${window.location.origin}${fileUrl}` : '';

  const handleCopyPath = useCallback(async () => {
    if (!fullPath) return;
    try {
      await navigator.clipboard.writeText(fullPath);
      setCopiedPath(true);
      setTimeout(() => setCopiedPath(false), 2000);
    } catch {
      console.error('复制地址失败');
    }
  }, [fullPath]);

  return (
    <div className="page-stack">
      <div className="card result-card">
        <div className="card-head">
          <span className="card-title">简历成品</span>
          <div className="card-head-actions">
            {isStreaming && (
              <span className="streaming-hint">
                {condensed && !polishedText ? '精简中' : '生成中'} · {liveSeconds}s
                <span className="dot-pulse" />
              </span>
            )}
            {elapsed != null && !isStreaming && (
              <span className="pipeline-elapsed">{elapsed}s</span>
            )}
            {polishedText && (
              <button type="button" className="ghost-button small" onClick={handleCopy}>
                {copied ? '已复制 ✓' : '复制全文'}
              </button>
            )}
          </div>
        </div>
        {thoughts.length > 0 && (
          <details className="thought-panel">
            <summary>思考过程（{thoughts.length}）</summary>
            {thoughts.map((t, i) => (
              <p key={i} className="thought-item">{t}</p>
            ))}
          </details>
        )}
        <div className="card-body result-body">
          {polishedText ? (
            <pre className="result-text">{polishedText}</pre>
          ) : (
            <div className="empty-state">
              <span className="empty-icon">✨</span>
              <p className="empty-title">
                {isStreaming ? '正在生成成品简历…' : '成品简历将在这里流式输出'}
              </p>
              <p className="empty-sub">在上方输入简历内容，点击「生成简历」</p>
            </div>
          )}
        </div>
      </div>

      {sourceText.trim() && (
        <details className="card original-card">
          <summary>原文对照</summary>
          <div className="card-body">
            <pre className="result-text original-text">{sourceText}</pre>
          </div>
        </details>
      )}

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
              <>
                <div className="doc-entry">
                  <span className="doc-icon">📄</span>
                  <div className="doc-info">
                    <span className="doc-name">{exportState.filename}</span>
                    <span className="doc-size">
                      {Math.max(1, Math.round((exportState.size ?? 0) / 1024))} KB
                    </span>
                  </div>
                  <div className="doc-actions">
                    <a
                      className="export-download"
                      href={fileUrl}
                      download={exportState.filename || '简历_优化版.docx'}
                    >
                      下载 .docx
                    </a>
                    <a
                      className="doc-open"
                      href={fileUrl}
                      target="_blank"
                      rel="noreferrer"
                    >
                      打开文件
                    </a>
                  </div>
                </div>
                <div className="doc-path-row">
                  <span className="doc-path-label">文件地址</span>
                  <a
                    className="doc-path"
                    href={fileUrl}
                    target="_blank"
                    rel="noreferrer"
                    title={fullPath}
                  >
                    {fullPath}
                  </a>
                  <button type="button" className="doc-path-copy" onClick={handleCopyPath}>
                    {copiedPath ? '已复制 ✓' : '复制'}
                  </button>
                </div>
              </>
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

      {exportState.status === 'ready' && exportState.previewUrl && (
        <div className="card preview-card">
          <div className="card-head">
            <span className="card-title">文件预览</span>
            <span className="preview-hint">只读预览，需要修改请下载后编辑</span>
          </div>
          <div className="card-body preview-body">
            <iframe
              className="doc-preview-frame"
              src={exportState.previewUrl}
              title="简历文档预览"
              sandbox=""
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default GeneratePage;
