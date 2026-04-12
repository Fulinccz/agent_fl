import type { AgentRequest, UploadRequest } from '../types';

const baseUrl = (import.meta.env.VITE_API_URL as string) || '/api';

class ApiClient {
  private baseURL: string;

  constructor(baseURL: string = baseUrl) {
    this.baseURL = baseURL;
  }

  async streamAgent(
    request: AgentRequest,
    signal?: AbortSignal,
    onEvent?: (event: StreamEvent) => void,
    onComplete?: () => void
  ): Promise<void> {
    console.log('=== ApiClient.streamAgent 开始 ===');
    
    const url = `${this.baseURL}/agent/stream`;
    console.log('📤 发送请求到:', url);
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: request.query,
          model: request.model || 'model_serving/Qwen3___5-0___8B',
          deepThinking: request.deepThinking || false
        }),
        signal
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No reader available');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              
              if (onEvent) {
                onEvent(data as StreamEvent);
              }
            } catch (e) {
              // 忽略非JSON行
            }
          }
        }
      }

      if (onComplete) {
        onComplete();
      }
      
      console.log('✅ 流式响应完成');
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('⚠️ 请求被中止');
        return;
      }
      throw error;
    }
  }

  async uploadFile(
    request: UploadRequest,
    signal?: AbortSignal
  ): Promise<{ response: string }> {
    const formData = new FormData();
    formData.append('file', request.file);
    formData.append('query', request.query);
    formData.append('provider', request.provider || 'local');

    const response = await fetch(`${this.baseURL}/agent/upload`, {
      method: 'POST',
      body: formData,
      signal
    });

    if (!response.ok) {
      let errorMessage = `Upload failed with status: ${response.status}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMessage = errorData.detail;
        }
      } catch (e) {
        // 如果解析JSON失败，使用默认错误信息
      }
      throw new Error(errorMessage);
    }

    return response.json();
  }

  async uploadFileStreaming(
    request: UploadRequest,
    signal?: AbortSignal,
    onChunk?: (content: string) => void,
    onError?: (error: string) => void,
    onComplete?: () => void,
    onThought?: (content: string) => void
  ): Promise<void> {
    const formData = new FormData();
    formData.append('file', request.file);
    formData.append('query', request.query);
    formData.append('provider', request.provider || 'local');

    try {
      const response = await fetch(`${this.baseURL}/agent/upload_stream`, {
        method: 'POST',
        body: formData,
        signal
      });

      if (!response.ok) {
        let errorMessage = `Upload failed with status: ${response.status}`;
        try {
          const errorData = await response.json();
          if (errorData.detail) {
            errorMessage = errorData.detail;
          }
        } catch (e) {
          // 如果解析 JSON 失败，使用默认错误信息
        }
        throw new Error(errorMessage);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Readable stream not supported');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // 处理可能的多个 JSON 对象
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // 保留最后一个不完整的行

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            // 处理流式事件
            if (data.type === 'token' && data.content) {
              onChunk?.(data.content);
            } else if (data.type === 'thought' && data.content) {
              onThought?.(data.content);
            } else if (data.type === 'error' && data.content) {
              onError?.(data.content);
            } else if (data.error) {
              onError?.(data.error);
            }
          } catch (e) {
            console.error('Failed to parse stream chunk:', e);
          }
        }
      }

      onComplete?.();
    } catch (error) {
      if (error instanceof Error && (error.name === 'AbortError' || error.message.includes('aborted'))) {
        // 用户取消，不处理
        return;
      }
      console.error('Streaming upload error:', error);
      onError?.(error instanceof Error ? error.message : 'Unknown error');
      throw error;
    }
  }

}

export interface StreamEvent {
  type: 'thought' | 'token' | 'complete' | 'error';
  content?: string;
  full_text?: string;
}

export const apiClient = new ApiClient();
export default ApiClient;
