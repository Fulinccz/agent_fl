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
    const imageUrl = loadImage(src, isCritical);
    
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