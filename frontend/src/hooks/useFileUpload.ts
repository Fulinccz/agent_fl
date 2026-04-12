import { useState, useCallback } from 'react';
import { agentService } from '../services/agentService';

interface UseFileUploadReturn {
  uploadedFile: File | null;
  fileName: string;
  isUploading: boolean;
  uploadError: string | null;
  handleFileSelect: (event: React.ChangeEvent<HTMLInputElement>) => void;
  uploadFile: (query: string, signal?: AbortSignal) => Promise<string>;
  clearFile: () => void;
}

export function useFileUpload(): UseFileUploadReturn {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      const validTypes = ['.pdf', '.docx'];
      const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
      
      if (!validTypes.includes(fileExtension)) {
        setUploadError('仅支持 .docx 和 .pdf 格式的文件');
        event.target.value = ''; // 清空文件选择
        return;
      }
      
      setUploadedFile(file);
      setUploadError(null);
    }
  }, []);

  const uploadFile = useCallback(async (
    query: string,
    signal?: AbortSignal
  ): Promise<string> => {
    if (!uploadedFile) throw new Error('No file selected');

    setIsUploading(true);
    setUploadError(null);

    try {
      const result = await agentService.uploadAndProcess(
        { file: uploadedFile, query },
        signal
      );
      
      setIsUploading(false);
      return result;
    } catch (error) {
      setIsUploading(false);
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      setUploadError(errorMessage);
      throw error;
    }
  }, [uploadedFile]);

  const clearFile = useCallback(() => {
    setUploadedFile(null);
    setUploadError(null);
  }, []);

  return {
    uploadedFile,
    fileName: uploadedFile?.name || '',
    isUploading,
    uploadError,
    handleFileSelect,
    uploadFile,
    clearFile
  };
}

export default useFileUpload;
