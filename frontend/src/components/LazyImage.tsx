import React, { useState, useEffect } from 'react';
import { loadImage } from '../utils/resourceLoader';

interface LazyImageProps {
  src: string;
  alt: string;
  isCritical?: boolean;
  placeholderSrc?: string;
  className?: string;
}

const LazyImage: React.FC<LazyImageProps> = ({
  src,
  alt,
  isCritical = false,
  placeholderSrc = '/src/assets/readyInClient/react.svg',
  className = ''
}) => {
  const [imageSrc, setImageSrc] = useState(placeholderSrc);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // 检查 src 是否为导入的模块（通常以 http 或 / 开头的是路径，否则是导入的模块）
    const isImportedModule = !src.startsWith('http') && !src.startsWith('/');
    
    const imageUrl = isImportedModule ? src : loadImage(src, isCritical);
    
    const img = new Image();
    img.src = imageUrl;
    
    img.onload = () => {
      setImageSrc(imageUrl);
      setIsLoaded(true);
    };
  }, [src, isCritical]);

  return (
    <img
      src={imageSrc}
      alt={alt}
      className={`lazy-image ${isLoaded ? 'loaded' : 'loading'} ${className}`}
      loading="lazy"
    />
  );
};

export default LazyImage;