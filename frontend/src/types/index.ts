export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

/** 页面入口：决定后端注入的提示词、任务计划与可用执行器 */
export type PageKey = 'analyze' | 'generate' | 'interview';

export interface UploadRequest {
  file: File;
  /** 用户补充要求（可选，作为润色附加指令） */
  query?: string;
  /** 目标岗位 JD（可选） */
  jd?: string;
  provider?: string;
  model?: string;
  /** 快捷操作显式意图（可选）：interview-prep / resume-score / jd-keyword-match */
  intent?: string;
  /** 页面入口（analyze / generate / interview） */
  page?: PageKey;
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
  /** 快捷操作显式意图（可选）：interview-prep / resume-score / jd-keyword-match，不传走简历全流程优化 */
  intent?: string;
  /** 页面入口（analyze / generate / interview） */
  page?: PageKey;
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

/** 导出结果：可直接打开/下载的地址 + 只读预览地址 */
export interface ExportDocxResult {
  url: string;
  preview_url: string;
  filename: string;
  size: number;
}

// ---- 面试攻略（整体攻略 + 高频问题）----

export interface InterviewStrategy {
  /** 候选人整体画像与面试主攻方向 */
  summary?: string;
  /** 核心亮点 */
  highlights?: string[];
  /** 主要风险 / 短板 */
  risks?: string[];
  /** 备战建议 */
  tips?: string[];
}

export interface InterviewQuestion {
  question: string;
  /** 所属考察维度 */
  category?: string;
  /** 考察点 */
  assess_point?: string;
  /** 答题要点 */
  answer_points?: string[];
  /** 可能追问 */
  follow_up?: string;
}

export interface InterviewGuideData {
  /** 目标岗位名称 */
  position?: string;
  /** llm（模型生成）| rule（规则兜底） */
  source?: string;
  strategy?: InterviewStrategy;
  /** 最可能被问的问题（按可能性排序） */
  questions?: InterviewQuestion[];
}

export interface InterviewData {
  interview_guide?: InterviewGuideData;
}

export interface CompleteData {
  elapsed_seconds?: number;
  [key: string]: unknown;
}

/** 上传链路解析出的简历正文（回填三页共用的输入框） */
export interface ParsedResumeData {
  text?: string;
}

/** 简历与 JD 的匹配分析（有 JD 时才有实质内容） */
export interface MatchAnalysis {
  match_score?: number;
  matched_keywords?: string[];
  missing_keywords?: string[];
  [key: string]: unknown;
}

export interface SuggestionItem {
  priority?: number;
  category?: string;
  suggestion?: string;
  example?: string;
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
  | 'interview'
  | 'resume_parsed'
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
