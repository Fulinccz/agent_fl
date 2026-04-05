import React from 'react';

interface ThoughtProcessProps {
  thoughts: string[];
  isStopped?: boolean;
}

const ThoughtProcess: React.FC<ThoughtProcessProps> = ({
  thoughts,
  isStopped = false
}) => {
  const displayContent = isStopped
    ? thoughts.length > 0 
      ? thoughts.join('\n') + '\n\n[用户已中止生成]'
      : '[用户已中止生成]'
    : thoughts.join('\n');

  return (
    <div className="output-section thought-section">
      <h3 className="output-title">💭 思考过程</h3>
      <div className="thought-box">
        {displayContent ? (
          <pre className="thought-content">{displayContent}</pre>
        ) : (
          <div className="thought-empty">等待输入...</div>
        )}
      </div>
    </div>
  );
};

export default ThoughtProcess;
