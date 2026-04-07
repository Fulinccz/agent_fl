import React, { useMemo } from 'react';

interface ChatOutputProps {
  content: string;
  title?: string;
  className?: string;
}

function formatContent(content: string): string {
  const trimmed = content.trim();
  
  if (!trimmed) return '';
  
  let result = trimmed;
  
  result = result.replace(/【简历评分】\s*/g, '\n【简历评分】\n');
  result = result.replace(/【核心提炼】\s*/g, '\n\n【核心提炼】\n');
  result = result.replace(/【优化建议】\s*/g, '\n\n【优化建议】\n');
  result = result.replace(/【优化结果】\s*/g, '\n\n【优化结果】\n');
  
  result = result.replace(/\n{3,}/g, '\n\n');
  
  return result.trim();
}

const ChatOutput: React.FC<ChatOutputProps> = ({
  content,
  title,
  className = ''
}) => {
  const formatted = useMemo(() => formatContent(content), [content]);
  
  if (!content.trim()) {
    return (
      <div className={`output-section ${className}`}>
        {title && <h3 className="output-title">{title}</h3>}
        <div className="output-content">
          <div className="output-empty">等待输入...</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`output-section ${className}`}>
      {title && <h3 className="output-title">{title}</h3>}
      <div className="output-content">
        <pre className="output-text">{formatted}</pre>
      </div>
    </div>
  );
};

export default ChatOutput;