import type {
  SkillRouteResponse,
  SkillExecuteAutoResponse,
  SkillExecuteRequest,
  SkillExecuteResponse,
  SkillListResponse
} from '../types/skill';

const baseUrl = (import.meta.env.VITE_API_URL as string) || '/api';

// ==================== Skill Router API ====================

class SkillApiClient {
  private baseURL: string;

  constructor(baseURL: string = baseUrl) {
    this.baseURL = baseURL;
  }

  /**
   * 意图路由（仅识别，不执行）
   * 
   * 返回路由决策信息，用于前端展示当前识别到的意图
   */
  async routeIntent(userInput: string): Promise<SkillRouteResponse> {
    const response = await fetch(
      `${this.baseURL}/skill/route?user_input=${encodeURIComponent(userInput)}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Intent routing failed');
    }

    return response.json();
  }

  /**
   * 自动识别并执行技能
   * 
   * 返回包含路由决策信息和执行结果
   */
  async executeAuto(userInput: string): Promise<SkillExecuteAutoResponse> {
    const response = await fetch(
      `${this.baseURL}/skill/execute/auto?user_input=${encodeURIComponent(userInput)}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Auto skill execution failed');
    }

    return response.json();
  }

  /**
   * 执行指定技能
   */
  async execute(request: SkillExecuteRequest): Promise<SkillExecuteResponse> {
    const response = await fetch(`${this.baseURL}/skill/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Skill execution failed');
    }

    return response.json();
  }

  /**
   * 列出所有可用技能
   */
  async listSkills(): Promise<SkillListResponse> {
    const response = await fetch(`${this.baseURL}/skill/list`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'List skills failed');
    }

    return response.json();
  }
}

// ==================== 导出实例 ====================

export const skillApiClient = new SkillApiClient();
export default SkillApiClient;
