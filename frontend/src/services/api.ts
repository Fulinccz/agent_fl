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
      throw new Error(`Upload failed with status: ${response.status}`);
    }

    return response.json();
  }
}

export interface StreamEvent {
  type: 'thought' | 'token' | 'complete' | 'error';
  content?: string;
  full_text?: string;
}

export const apiClient = new ApiClient();
export default ApiClient;
