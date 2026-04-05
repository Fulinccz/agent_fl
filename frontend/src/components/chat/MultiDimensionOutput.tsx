import React from 'react';

interface DimensionData {
  parse?: any;
  score?: any;
  optimize?: any;
  polish?: any;
}

interface MultiDimensionOutputProps {
  data: DimensionData;
  className?: string;
}

const MultiDimensionOutput: React.FC<MultiDimensionOutputProps> = ({
  data,
  className = ''
}) => {
  const formatJson = (obj: any): string => {
    if (!obj) return '暂无数据';
    try {
      return JSON.stringify(obj, null, 2);
    } catch {
      return String(obj);
    }
  };

  const dimensions = [
    { key: 'parse', title: '📋 简历解析', data: data.parse },
    { key: 'score', title: '📊 评分结果', data: data.score },
    { key: 'optimize', title: '✨ 优化建议', data: data.optimize },
    { key: 'polish', title: '✍️ 润色结果', data: data.polish },
  ];

  return (
    <div className={`multi-dimension-output ${className}`}>
      {dimensions.map(({ key, title, data: dimData }) => (
        <div key={key} className="dimension-section">
          <h4 className="dimension-title">{title}</h4>
          <div className="dimension-content">
            {dimData ? (
              <pre className="json-output">{formatJson(dimData)}</pre>
            ) : (
              <span className="dimension-empty">等待分析...</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default MultiDimensionOutput;