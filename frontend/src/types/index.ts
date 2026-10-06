export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

export interface UploadRequest {
  file: File;
  query: string;
  provider?: string;
}

// ==================== 简历优化多 Agent 接口 ====================

export interface ResumeOptimizeRequest {
  resume: string;
  jd?: string;
  position_type?: string;
  /** 模型提供者：local（本地 Qwen）| deepseek（DeepSeek OpenAPI），不传用服务端默认 */
  provider?: string;
  /** 模型名称，不传用该提供者默认 */
  model?: string;
}

export interface ResumeOptimizeEvent {
  type: 'score' | 'suggestions' | 'polished' | 'complete' | 'error';
  data?: any;
  message?: string;
}


