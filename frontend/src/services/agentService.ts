import { resumeOptimizeApiClient, uploadApiClient } from './api';
import type { ResumeOptimizeRequest, UploadRequest, ResumeOptimizeEvent } from '../types';

/**
 * AgentService - 简历优化服务
 * 
 * 使用多 Agent 工作流进行简历优化：
 * 1. ResumeScoreAgent - 简历评分
 * 2. JDMatchAgent - JD 关键词匹配与优化建议
 * 3. ResumePolishAgent - 简历润色
 */
class AgentService {
  /**
   * 流式简历优化
   * 
   * @param request 简历优化请求
   * @param signal 用于取消请求的 AbortSignal
   * @param onScore 评分结果回调
   * @param onSuggestions 优化建议回调
   * @param onPolished 润色结果回调
   * @param onComplete 完成回调
   * @param onError 错误回调
   */
  async optimizeResumeStream(
    request: ResumeOptimizeRequest,
    signal?: AbortSignal,
    onScore?: (data: { overall_score: any; scores: any }) => void,
    onSuggestions?: (data: { suggestions: any; match_analysis: any }) => void,
    onPolished?: (data: { optimized_resume: string }) => void,
    onComplete?: (data: any) => void,
    onError?: (error: string) => void
  ): Promise<void> {
    return resumeOptimizeApiClient.optimizeStream(
      request,
      signal,
      (event: ResumeOptimizeEvent) => {
        switch (event.type) {
          case 'score':
            if (onScore && event.data) onScore(event.data);
            break;
          case 'suggestions':
            if (onSuggestions && event.data) onSuggestions(event.data);
            break;
          case 'polished':
            if (onPolished && event.data) onPolished(event.data);
            break;
          case 'complete':
            if (onComplete && event.data) onComplete(event.data);
            break;
          case 'error':
            if (onError && event.message) onError(event.message);
            break;
        }
      },
      () => {
        // onComplete 已经在上面处理了
      }
    );
  }

  /**
   * 非流式简历优化
   * 
   * @param request 简历优化请求
   * @returns 优化结果
   */
  async optimizeResume(request: ResumeOptimizeRequest) {
    return resumeOptimizeApiClient.optimize(request);
  }

  /**
   * 上传文件并处理
   * 
   * @param request 上传请求
   * @param signal 用于取消请求的 AbortSignal
   * @returns 处理结果
   */
  async uploadAndProcess(
    request: UploadRequest,
    signal?: AbortSignal
  ): Promise<string> {
    const result = await uploadApiClient.uploadFile(request, signal);
    return result.response;
  }
}

export const agentService = new AgentService();
export default AgentService;
