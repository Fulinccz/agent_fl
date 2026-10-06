// ==================== Skill Router 接口 ====================

/**
 * 路由决策信息
 */
export interface SkillRouteInfo {
  /** 识别到的技能名称 */
  skill: string;
  /** 置信度 (0-1) */
  confidence: number;
  /** 决策来源: keyword | vector | llm */
  source: string;
  /** 决策原因说明 */
  reason: string;
  /** 额外参数 */
  params?: Record<string, any>;
}

/**
 * 意图路由请求
 */
export interface SkillRouteRequest {
  user_input: string;
}

/**
 * 意图路由响应
 */
export interface SkillRouteResponse {
  skill: string;
  confidence: number;
  source: string;
  reason: string;
  params?: Record<string, any>;
  error?: string;
}

/**
 * 技能执行请求
 */
export interface SkillExecuteRequest {
  skill_name: string;
  parameters?: Record<string, any>;
}

/**
 * 技能执行响应
 */
export interface SkillExecuteResponse {
  result?: any;
  error?: string;
}

/**
 * 自动执行响应（包含路由信息）
 */
export interface SkillExecuteAutoResponse {
  /** 路由决策信息 */
  route: SkillRouteInfo;
  /** 技能执行结果 */
  result: any;
  error?: string;
}

/**
 * 技能列表响应
 */
export interface SkillListResponse {
  skills: string[];
  error?: string;
}

/**
 * 技能路由状态（用于前端展示）
 */
export interface SkillRouteState {
  /** 当前识别到的技能 */
  currentSkill: string | null;
  /** 置信度 */
  confidence: number;
  /** 决策来源 */
  source: string;
  /** 原因说明 */
  reason: string;
  /** 是否正在路由中 */
  isRouting: boolean;
  /** 路由错误 */
  error: string | null;
}
