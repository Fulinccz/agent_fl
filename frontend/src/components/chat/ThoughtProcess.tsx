import React from 'react';

interface ThoughtProcessProps {
  thoughts: string[];
  isStopped?: boolean;
}

const ThoughtProcess: React.FC<ThoughtProcessProps> = ({
  thoughts,
  isStopped = false
}) => {
  const displayThoughts = isStopped
    ? thoughts.length > 0 
      ? [...thoughts, '用户已中止生成']
      : ['用户已中止生成']
    : thoughts;

  return (
    <div className="output-section thought-section">
      <h3 className="output-title">思考过程</h3>
      <div className="thought-box">
        {displayThoughts.length > 0 ? (
          displayThoughts.map((thought, index) => (
            <div key={index} className="thought-item">
              {thought}
            </div>
          ))
        ) : (
          <div className="thought-item" style={{ opacity: 0.5 }}>等待输入...</div>
        )}
      </div>
    </div>
  );
};

export default ThoughtProcess;
