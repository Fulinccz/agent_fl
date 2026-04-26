import { useState, useCallback, useRef } from 'react';
import type { ResumeOptimizeRequest } from '../types';
import { agentService } from '../services/agentService';

interface UseAgentOptions {
  onScore?: (data: { overall_score: any; scores: any }) => void;
  onSuggestions?: (data: { suggestions: any; match_analysis: any }) => void;
  onPolished?: (data: { optimized_resume: string }) => void;
  onComplete?: (data: any) => void;
  onError?: (error: string) => void;
}

interface UseAgentReturn {
  isLoading: boolean;
  isStopped: boolean;
  error: string | null;
  execute: (request: ResumeOptimizeRequest, signal?: AbortSignal) => Promise<void>;
  stop: () => void;
  reset: () => void;
}

export function useAgent(options: UseAgentOptions = {}): UseAgentReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [isStopped, setIsStopped] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const abortControllerRef = useRef<AbortController | null>(null);

  const execute = useCallback(async (
    request: ResumeOptimizeRequest,
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
      await agentService.optimizeResumeStream(
        request,
        controller.signal,
        options.onScore,
        options.onSuggestions,
        options.onPolished,
        (data) => {
          if (options.onComplete) options.onComplete(data);
          setIsLoading(false);
        },
        (err) => {
          if (options.onError) options.onError(err);
          setError(err);
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
