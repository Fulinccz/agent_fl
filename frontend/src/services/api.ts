import type {
  UploadRequest,
  ResumeOptimizeRequest,
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
    signal?: AbortSignal,
    onToken?: (token: string) => void
  ): Promise<{ response: string }> {
    const formData = new FormData();
    formData.append('file', request.file);
    formData.append('query', request.query);
    if (request.provider) formData.append('provider', request.provider);
    if ((request as any).model) formData.append('model', (request as any).model);

    // 后端 /agent/upload_stream 返回 JSON-lines 流（每行一个事件对象）
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
        // 如果解析JSON失败，使用默认错误信息
      }
      throw new Error(errorMessage);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('响应不支持流式读取');

    const decoder = new TextDecoder();
    let buffer = '';
    let fullText = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // 保留未完成的残行

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const event = JSON.parse(trimmed);
          if (event.type === 'token' && event.content) {
            fullText += event.content;
            if (onToken) onToken(event.content);
          } else if (event.type === 'error') {
            throw new Error(event.message || event.content || '服务器处理出错');
          }
          // complete 等其他事件：文本已由 token 累积
        } catch (e) {
          if (e instanceof SyntaxError) continue; // 非法 JSON 行跳过
          throw e;
        }
      }
    }

    return { response: fullText };
  }
}

// ==================== 简历优化多 Agent API ====================

export class ResumeOptimizeApiClient {
  private baseURL: string;

  constructor(baseURL: string = baseUrl) {
    this.baseURL = baseURL;
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
