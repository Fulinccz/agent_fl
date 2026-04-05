import { useState, useCallback, useRef } from 'react';
import type { AgentRequest } from '../types';
import { agentService } from '../services/agentService';

interface UseAgentOptions {
  onThought?: (content: string) => void;
  onContent?: (content: string) => void;
  onError?: (error: string) => void;
  onComplete?: () => void;
}

interface UseAgentReturn {
  isLoading: boolean;
  isStopped: boolean;
  error: string | null;
  execute: (request: AgentRequest, signal?: AbortSignal) => Promise<void>;
  stop: () => void;
  reset: () => void;
}

export function useAgent(options: UseAgentOptions = {}): UseAgentReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [isStopped, setIsStopped] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const abortControllerRef = useRef<AbortController | null>(null);

  const execute = useCallback(async (
    request: AgentRequest,
    signal?: AbortSignal
  ) => {
    setIsLoading(true);
    setIsStopped(false);
    setError(null);

    const controller = signal ? 
      AbortController.prototype : 
      new AbortController();
    
    if (!signal) {
      abortControllerRef.current = controller;
    }

    try {
      await agentService.streamQuery(
        request,
        controller.signal,
        options.onThought,
        options.onContent,
        options.onError,
        () => {
          if (options.onComplete) options.onComplete();
          setIsLoading(false);
        }
      );
    } catch (err) {
      if (err instanceof Error && err.name !== 'AbortError') {
        const errorMessage = err.message || 'An error occurred';
        setError(errorMessage);
        if (options.onError) options.onError(errorMessage);
      }
      setIsLoading(false);
    }
  }, [options]);

  const stop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsStopped(true);
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setError(null);
    setIsStopped(false);
  }, []);

  return {
    isLoading,
    isStopped,
    error,
    execute,
    stop,
    reset
  };
}

export default useAgent;
