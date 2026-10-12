import React from 'react';

interface ChatInputProps {
  onSubmit: () => void;
  onStop: () => void;
  isLoading: boolean;
  uploadedFile: File | null;
  onFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onFileClear?: () => void;
  disabled?: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSubmit,
  onStop,
  isLoading,
  uploadedFile,
  onFileChange,
  onFileClear,
  disabled = false
}) => {
  const blocked = disabled || isLoading;

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
              accept=".pdf,.doc,.docx,.txt,.png,.jpg,.jpeg,.bmp,.webp,.mp3,.wav,.m4a,.flac,.ogg"
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
          disabled={disabled}
          title="发送（Enter）"
        >
          开始优化 ↑
        </button>
      )}
    </div>
  );
};

export default ChatInput;
