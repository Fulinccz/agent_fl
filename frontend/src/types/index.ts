export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

export interface UploadRequest {
  file: File;
  /** 用户补充要求（可选，作为润色附加指令） */
  query?: string;
  /** 目标岗位 JD（可选） */
  jd?: string;
  provider?: string;
  model?: string;
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

// ---- SSE 事件负载类型（后端字段宽松，前端按结构消费）----

export interface ScoreOverall {
  score?: number;
  rating?: string;
  description?: string;
}

export interface ScoreData {
  overall_score?: ScoreOverall;
  scores?: Record<string, number>;
  dimension_analysis?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface SuggestionsData {
  suggestions?: unknown;
  match_analysis?: unknown;
  [key: string]: unknown;
}

export interface PolishedData {
  optimized_resume: string;
  /** true 表示流式中的增量快照（前端节流渲染） */
  partial?: boolean;
}

export interface CondensedData {
  condensed_resume: string;
  original_length?: number;
  condensed_length?: number;
}

export interface CompleteData {
  elapsed_seconds?: number;
  [key: string]: unknown;
}

export interface StepStartData {
  step_id: string;
  agent: string;
  task: string;
}

export interface StepResultData {
  step_id: string;
  agent: string;
  result?: unknown;
}

export type ResumeEventType =
  | 'score'
  | 'suggestions'
  | 'polished'
  | 'condensed'
  | 'complete'
  | 'error'
  | 'intent'
  | 'plan'
  | 'step_start'
  | 'step_result'
  | 'reflection'
  | 'thought';

export interface ResumeOptimizeEvent {
  type: ResumeEventType;
  // 不同 type 对应不同负载，消费处按 type 收窄
  data?: unknown;
  message?: string;
  // thought 事件负载（ReAct 步内思考）
  content?: string;
}
