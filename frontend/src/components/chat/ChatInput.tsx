import React from 'react';

interface ChatInputProps {
  onSubmit: () => void;
  onStop: () => void;
  isLoading: boolean;
  uploadedFile: File | null;
  onFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onFileClear?: () => void;
  disabled?: boolean;
  /** 语音输入：录音中 / 转写中 / 当前秒数 */
  isRecording?: boolean;
  isTranscribing?: boolean;
  recordSeconds?: number;
  voiceSupported?: boolean;
  onVoiceToggle?: () => void;
  /** 提交按钮文案（各页面动作不同） */
  submitLabel?: string;
  /** 是否已有可提交内容（简历文本或附件）：无内容时禁用提交并提示原因 */
  hasInput?: boolean;
}

function formatSeconds(total: number): string {
  const mm = String(Math.floor(total / 60)).padStart(2, '0');
  const ss = String(total % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSubmit,
  onStop,
  isLoading,
  uploadedFile,
  onFileChange,
  onFileClear,
  disabled = false,
  isRecording = false,
  isTranscribing = false,
  recordSeconds = 0,
  voiceSupported = true,
  onVoiceToggle,
  submitLabel = '开始优化 ↑',
  hasInput = true
}) => {
  const blocked = disabled || isLoading;
  const voiceDisabled = blocked || !voiceSupported || isTranscribing;

  // 提交按钮被禁用的原因：无内容 / 其他页面生成中 / 语音未落定
  const submitTitle = !hasInput
    ? '请先填写简历或上传文件'
    : disabled
      ? '其他页面正在生成，请稍候'
      : isRecording || isTranscribing
        ? '语音处理中，请稍候'
        : '提交';

  const voiceTitle = !voiceSupported
    ? '当前浏览器不支持录音（需 localhost 或 HTTPS）'
    : isTranscribing
      ? '正在转写录音，请稍候'
      : isRecording
        ? '点击结束录音并转写（最长 60 秒）'
        : '语音输入：点击开始录音，转写为文字后可校对';

  const voiceClass = [
    'icon-button',
    'voice-button',
    isRecording ? 'recording' : '',
    isTranscribing ? 'transcribing' : '',
    voiceDisabled && !isRecording ? 'disabled' : ''
  ].filter(Boolean).join(' ');

  return (
    <div className="input-actions">
      <div className="actions-left">
        <div className="attachment-container">
          <label
            className={`icon-button${blocked ? ' disabled' : ''}`}
            title="支持文档 / 图片 / 音频（.pdf .doc .docx .txt .png .jpg .mp3 .wav 等）"
          >
            <input
              type="file"
              onChange={onFileChange}
              className="file-upload"
              accept=".pdf,.doc,.docx,.txt,.png,.jpg,.jpeg,.bmp,.webp,.mp3,.wav,.m4a,.flac,.ogg,.webm"
              disabled={blocked}
            />
            📎
          </label>
          {uploadedFile && (
            <div className="file-chip">
              <span className="file-name">{uploadedFile.name}</span>
              <button
                type="button"
                className="remove-file"
                onClick={() => onFileClear?.()}
                title="移除文件"
              >
                ✕
              </button>
            </div>
          )}
        </div>

        <button
          type="button"
          className={voiceClass}
          title={voiceTitle}
          onClick={onVoiceToggle}
          disabled={voiceDisabled}
        >
          {isTranscribing ? (
            <span className="voice-transcribing">转写中</span>
          ) : isRecording ? (
            <>
              <span className="voice-dot" />
              {formatSeconds(recordSeconds)}
            </>
          ) : (
            '🎤'
          )}
        </button>
      </div>

      {isLoading ? (
        <button type="button" className="send-button stop" onClick={onStop} title="停止生成">
          ■ 停止
        </button>
      ) : (
        <button
          type="submit"
          className="send-button"
          onClick={onSubmit}
          disabled={disabled || !hasInput || isRecording || isTranscribing}
          title={submitTitle}
        >
          {submitLabel}
        </button>
      )}
    </div>
  );
};

export default ChatInput;
