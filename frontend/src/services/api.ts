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

  /**
   * 上传文件并进入 MAS 流水线
   *
   * 事件契约与 /resume/optimize/stream 完全一致（score / suggestions / condensed /
   * polished / step 过程事件 / complete / error），由调用方统一消费。
   */
  async uploadFile(
    request: UploadRequest,
    signal?: AbortSignal,
    onEvent?: (event: ResumeOptimizeEvent) => void
  ): Promise<void> {
    const formData = new FormData();
    formData.append('file', request.file);
    if (request.query) formData.append('query', request.query);
    if (request.jd) formData.append('jd', request.jd);
    if (request.provider) formData.append('provider', request.provider);
    if (request.model) formData.append('model', request.model);

    try {
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
          // 后端上传接口错误体为 {error}，统一异常处理器为 {message}/{detail}
          errorMessage = errorData.error || errorData.message || errorData.detail || errorMessage;
        } catch {
          // 解析 JSON 失败时使用默认错误信息
        }
        throw new Error(errorMessage);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('响应不支持流式读取');

      const decoder = new TextDecoder();
      let buffer = '';

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
            const event = JSON.parse(trimmed) as ResumeOptimizeEvent;
            if (event.type === 'error') {
              throw new Error(event.message || '服务器处理出错');
            }
            onEvent?.(event);
          } catch (e) {
            if (e instanceof SyntaxError) continue; // 非法 JSON 行跳过
            throw e;
          }
        }
      }
    } catch (error) {
      // 用户主动中止：静默结束，与优化流保持一致
      if (error instanceof Error && error.name === 'AbortError') return;
      throw error;
    }
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
    const url = `${this.baseURL}/resume/optimize/stream`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal
      });

      if (!response.ok) {
        let errorMessage = `HTTP error! status: ${response.status}`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.error || errorData.message || errorData.detail || errorMessage;
        } catch {
          // 解析失败用默认信息
        }
        throw new Error(errorMessage);
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
              if (onEvent) {
                onEvent(data);
              }
            } catch {
              console.warn('解析事件失败:', line);
            }
          }
        }
      }

      if (onComplete) {
        onComplete();
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        return;
      }
      throw error;
    }
  }

  /**
   * 导出优化简历 docx
   *
   * 双路径：携带 file（用户上传的原始 docx）→ 服务端按内容对齐改写；
   * 不携带 → 按内置模板重新排版生成。返回 blob 与后端建议文件名。
   */
  async exportDocx(text: string, file?: File | null): Promise<{ blob: Blob; filename: string }> {
    const formData = new FormData();
    formData.append('text', text);
    if (file) formData.append('file', file);

    // 导出可能涉及 docx 解析与重排，加 60s 超时避免弱网下永久 pending
    const response = await fetch(`${this.baseURL}/resume/export-docx`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(60_000)
    });

    if (!response.ok) {
      let errorMessage = `导出失败（${response.status}）`;
      try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorData.message || errorData.detail || errorMessage;
      } catch {
        // 解析失败用默认信息
      }
      throw new Error(errorMessage);
    }

    const blob = await response.blob();
    const disposition = response.headers.get('Content-Disposition') || '';
    let filename = '简历_优化版.docx';
    const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
    const plainMatch = disposition.match(/filename="?([^";]+)"?/i);
    if (utf8Match) {
      filename = decodeURIComponent(utf8Match[1]);
    } else if (plainMatch) {
      filename = plainMatch[1];
    }
    return { blob, filename };
  }
}

// ==================== 导出实例 ====================

export const uploadApiClient = new UploadApiClient();
export const resumeOptimizeApiClient = new ResumeOptimizeApiClient();
