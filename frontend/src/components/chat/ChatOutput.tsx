import React from 'react';

interface ChatOutputProps {
  content: string;
  title?: string;
  className?: string;
}

const ChatOutput: React.FC<ChatOutputProps> = ({
  content,
  title,
  className = ''
}) => {
  return (
    <div className={`output-section ${className}`}>
      {title && <h3 className="output-title">{title}</h3>}
      <div className="output-content">
        {content || ''}
      </div>
    </div>
  );
};

export default ChatOutput;
