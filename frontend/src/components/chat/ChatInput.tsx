import React, { useRef } from 'react';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  isLoading: boolean;
  uploadedFile: File | null;
  onFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onNewChat: () => void;
  disabled?: boolean;
  deepThinking?: boolean;
  onDeepThinkToggle?: () => void;
}

const ChatInput: React.FC<ChatInputProps> = ({
  value,
  onChange,
  onSubmit,
  onStop,
  isLoading,
  uploadedFile,
  onFileChange,
  onNewChat,
  disabled = false,
  deepThinking = false,
  onDeepThinkToggle
}) => {
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && !isLoading) {
        onSubmit();
      }
    }
  };

  return (
    <div className="input-container">
      <div className="input-wrapper">
        <textarea
          ref={inputRef}
          className="input"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="请发送消息~"
          disabled={disabled || isLoading}
          rows={3}
        />
      </div>
      
      <div className="form-actions">
        <div className="left-actions">
          <button 
            className="new-chat-button"
            onClick={onNewChat}
            title="开启新对话"
          >
            开启新对话
          </button>
          
          <button 
            className={`deep-think-button${deepThinking ? ' active' : ''}`}
            onClick={onDeepThinkToggle}
            title={deepThinking ? '关闭深度思考' : '开启深度思考'}
            style={{
              borderColor: deepThinking ? '#6eb5ff' : 'var(--border-color)',
              color: deepThinking ? '#6eb5ff' : 'var(--text-secondary)'
            }}
          >
            深度思考
          </button>
          
          <div className="attachment-container">
            <label 
              className={`attachment-button${isLoading ? ' disabled' : ''}`}
              style={{ opacity: isLoading ? 0.5 : 1, cursor: isLoading ? 'not-allowed' : 'pointer' }}
            >
              <input
                type="file"
                onChange={onFileChange}
                className="file-upload"
                accept=".pdf,.doc,.docx,.txt,.jpg,.png"
                disabled={isLoading}
              />
              📎
            </label>
            {uploadedFile && (
              <div className="file-info">
                <span className="file-name">{uploadedFile.name}</span>
                <button 
                  className="remove-file"
                  onClick={() => onFileChange({ target: { files: {} } } as any)}
                  title="移除文件"
                >
                  ✕
                </button>
              </div>
            )}
          </div>
        </div>

        {isLoading ? (
          <button 
            className="send-button"
            onClick={onStop}
            title="停止生成"
          >
            ■
          </button>
        ) : (
          <button 
            className="send-button"
            onClick={onSubmit}
            disabled={disabled || !value.trim()}
            title="发送消息"
          >
            ↑
          </button>
        )}
      </div>
    </div>
  );
};

export default ChatInput;