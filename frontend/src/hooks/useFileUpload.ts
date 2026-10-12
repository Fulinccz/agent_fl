import { useState, useCallback } from 'react';

interface UseFileUploadReturn {
  uploadedFile: File | null;
  fileName: string;
  handleFileSelect: (event: React.ChangeEvent<HTMLInputElement>) => void;
  clearFile: () => void;
}

/**
 * 文件选择 hook（仅负责选择与前端校验）。
 * 上传后的流式处理统一由 useStreamResponse.startUpload 走 MAS 事件流。
 */
export function useFileUpload(): UseFileUploadReturn {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      const validTypes = ['.pdf', '.doc', '.docx', '.txt', '.png', '.jpg', '.jpeg', '.bmp', '.webp', '.mp3', '.wav', '.m4a', '.flac', '.ogg'];
      const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();

      if (!validTypes.includes(fileExtension)) {
        window.alert('支持文档（.pdf/.doc/.docx/.txt）、图片（.png/.jpg 等）、音频（.mp3/.wav/.m4a）');
        event.target.value = ''; // 清空文件选择
        return;
      }

      setUploadedFile(file);
    }
  }, []);

  const clearFile = useCallback(() => {
    setUploadedFile(null);
  }, []);

  return {
    uploadedFile,
    fileName: uploadedFile?.name || '',
    handleFileSelect,
    clearFile
  };
}

export default useFileUpload;
