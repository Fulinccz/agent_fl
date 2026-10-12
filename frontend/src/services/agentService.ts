import { resumeOptimizeApiClient, uploadApiClient } from './api';
import type {
  ResumeOptimizeRequest,
  UploadRequest,
  ResumeOptimizeEvent,
} from '../types';

/**
 * AgentService - 简历优化服务
 *
 * 粘贴简历与上传文件两条链路共用同一套 MAS 事件契约
 * （score → jd_match → resume_condense? → resume_polish），
 * 这里只提供两个薄入口，事件分发由调用方（useStreamResponse）统一处理。
 */
class AgentService {
  /** 粘贴简历：流式优化 */
  async runOptimizeStream(
    request: ResumeOptimizeRequest,
    signal: AbortSignal | undefined,
    onEvent: (event: ResumeOptimizeEvent) => void
  ): Promise<void> {
    return resumeOptimizeApiClient.optimizeStream(request, signal, onEvent);
  }

  /** 上传文件：解析后进入同一套 MAS 流水线 */
  async runUploadStream(
    request: UploadRequest,
    signal: AbortSignal | undefined,
    onEvent: (event: ResumeOptimizeEvent) => void
  ): Promise<void> {
    return uploadApiClient.uploadFile(request, signal, onEvent);
  }

  /**
   * 导出优化简历 docx
   *
   * @param text 优化后的简历文本
   * @param file 用户上传的原始 docx（可选，有则按对齐改写路径）
   */
  async exportDocx(text: string, file?: File | null): Promise<{ blob: Blob; filename: string }> {
    return resumeOptimizeApiClient.exportDocx(text, file);
  }
}

export const agentService = new AgentService();
export default AgentService;
