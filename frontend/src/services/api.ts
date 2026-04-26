import type {
  UploadRequest,
  ResumeOptimizeRequest,
  ResumeOptimizeResponse,
  ResumeOptimizeEvent
} from '../types';

const baseUrl = (import.meta.env.VITE_API_URL as string) || '/api';

// ==================== 文件上传 API ====================

class UploadApiClient {
  private baseURL: string;

  constructor(baseURL: string = baseUrl) {
    this.baseURL = baseURL;
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
}

// ==================== 简历优化多 Agent API ====================

export class ResumeOptimizeApiClient {
  private baseURL: string;

  constructor(baseURL: string = baseUrl) {
    this.baseURL = baseURL;
  }

  /**
   * 非流式简历优化
   */
  async optimize(request: ResumeOptimizeRequest): Promise<ResumeOptimizeResponse> {
    const response = await fetch(`${this.baseURL}/resume/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Resume optimization failed');
    }

    return response.json();
  }

  /**
   * 流式简历优化
   * 
   * 事件类型：
   * - type="score": 评分结果 { overall_score, scores }
   * - type="suggestions": 优化建议 { suggestions, match_analysis }
   * - type="polished": 润色后的简历 { optimized_resume }
   * - type="complete": 全部完成 { overall_score, scores, suggestions, optimized_resume, match_analysis }
   * - type="error": 错误 { message }
   */
  async optimizeStream(
    request: ResumeOptimizeRequest,
    signal?: AbortSignal,
    onEvent?: (event: ResumeOptimizeEvent) => void,
    onComplete?: () => void
  ): Promise<void> {
    console.log('=== ResumeOptimizeApiClient.optimizeStream 开始 ===');
    
    const url = `${this.baseURL}/resume/optimize/stream`;
    console.log('📤 发送请求到:', url);
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
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
              const data = JSON.parse(line) as ResumeOptimizeEvent;
              
              console.log(`📥 收到事件: type=${data.type}`);
              
              if (onEvent) {
                onEvent(data);
              }
            } catch (e) {
              console.warn('解析事件失败:', line);
            }
          }
        }
      }

      if (onComplete) {
        onComplete();
      }
      
      console.log('✅ 流式优化完成');
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('⚠️ 请求被中止');
        return;
      }
      throw error;
    }
  }
}

// ==================== 导出实例 ====================

export const uploadApiClient = new UploadApiClient();
export const resumeOptimizeApiClient = new ResumeOptimizeApiClient();
