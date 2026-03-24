import React from 'react';
import LazyImage from './LazyImage';

const NonCriticalComponent: React.FC = () => {
  return (
    <div className="non-critical-component">
      <h2>Additional Features</h2>
      {/* 非关键图片通过CDN加载 */}
      <LazyImage 
        src="/src/assets/retrieveData/images/feature-1.jpg" 
        alt="Feature 1" 
        isCritical={false}
      />
      <LazyImage 
        src="/src/assets/retrieveData/images/feature-2.jpg" 
        alt="Feature 2" 
        isCritical={false}
      />
    </div>
  );
};

export default NonCriticalComponent;