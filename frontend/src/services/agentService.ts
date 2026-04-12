import { apiClient } from './api';
import type { AgentRequest, UploadRequest } from '../types';

class AgentService {
  async streamQuery(
    request: AgentRequest,
    signal?: AbortSignal,
    onThought?: (content: string) => void,
    onContent?: (content: string) => void,
    onError?: (error: string) => void,
    onComplete?: () => void
  ): Promise<void> {
    return apiClient.streamAgent(
      request,
      signal,
      (event) => {
        switch (event.type) {
          case 'thought':
            if (onThought && event.content) onThought(event.content);
            break;
          case 'token':
            if (onContent && event.content) onContent(event.content);
            break;
          case 'error':
            if (onError && event.content) onError(event.content);
            break;
          case 'complete':
            if (onComplete) onComplete();
            break;
        }
      },
      onComplete
    );
  }

  async uploadAndProcess(
    request: UploadRequest,
    signal?: AbortSignal
  ): Promise<string> {
    const result = await apiClient.uploadFile(request, signal);
    return result.response;
  }

  async uploadAndProcessStream(
    request: UploadRequest,
    signal?: AbortSignal,
    onContent?: (content: string) => void,
    onError?: (error: string) => void,
    onComplete?: () => void,
    onThought?: (content: string) => void
  ): Promise<void> {
    return apiClient.uploadFileStreaming(
      request,
      signal,
      (response) => {
        if (onContent) onContent(response);
      },
      (error) => {
        if (onError) onError(error);
      },
      onComplete,
      onThought
    );
  }
}

export const agentService = new AgentService();
export default AgentService;
