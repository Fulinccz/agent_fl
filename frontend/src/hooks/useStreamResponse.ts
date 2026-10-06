import { useState, useCallback, useRef } from 'react';
import { agentService } from '../services/agentService';
import type { ResumeOptimizeRequest } from '../types';

interface UseStreamResponseOptions {
  onScoreUpdate?: (score: { overall_score: any; scores: any }) => void;
  onSuggestionsUpdate?: (suggestions: { suggestions: any; match_analysis: any }) => void;
  onPolishedUpdate?: (polished: { optimized_resume: string }) => void;
  onStreamError?: (error: string) => void;
}

interface UseStreamResponseReturn {
  score: { overall_score: any; scores: any } | null;
  suggestions: { suggestions: any; match_analysis: any } | null;
  polished: { optimized_resume: string } | null;
  error: string | null;
  isStreaming: boolean;
  startStream: (request: ResumeOptimizeRequest, signal?: AbortSignal) => Promise<void>;
  applyPolished: (text: string) => void;
  clearOutput: () => void;
}

export function useStreamResponse(
  options: UseStreamResponseOptions = {}
): UseStreamResponseReturn {
  const [score, setScore] = useState<{ overall_score: any; scores: any } | null>(null);
  const [suggestions, setSuggestions] = useState<{ suggestions: any; match_analysis: any } | null>(null);
  const [polished, setPolished] = useState<{ optimized_resume: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  
  const scoreRef = useRef<{ overall_score: any; scores: any } | null>(null);
  const suggestionsRef = useRef<{ suggestions: any; match_analysis: any } | null>(null);
  const polishedRef = useRef<{ optimized_resume: string } | null>(null);

  const startStream = useCallback(async (
    request: ResumeOptimizeRequest,
    signal?: AbortSignal
  ) => {
    setScore(null);
    setSuggestions(null);
    setPolished(null);
    setError(null);
    scoreRef.current = null;
    suggestionsRef.current = null;
    polishedRef.current = null;
    setIsStreaming(true);

    try {
      await agentService.optimizeResumeStream(
        request,
        signal,
        (data) => {
          scoreRef.current = data;
          setScore(data);
          if (options.onScoreUpdate) {
            options.onScoreUpdate(data);
          }
        },
        (data) => {
          suggestionsRef.current = data;
          setSuggestions(data);
          if (options.onSuggestionsUpdate) {
            options.onSuggestionsUpdate(data);
          }
        },
        (data) => {
          // 避免重复设置相同内容
          const currentText = polishedRef.current?.optimized_resume || '';
          const newText = data?.optimized_resume || '';

          // 只有当内容真正变化时才更新
          if (newText !== currentText) {
            polishedRef.current = data;
            setPolished(data);
            if (options.onPolishedUpdate) {
              options.onPolishedUpdate(data);
            }
          }
        },
        undefined,
        (msg: string) => {
          // 后端 error 事件：上屏给用户
          setError(msg);
          if (options.onStreamError) {
            options.onStreamError(msg);
          }
        }
      );
    } catch (e: any) {
      const msg = e?.message || '请求失败';
      setError(msg);
      if (options.onStreamError) {
        options.onStreamError(msg);
      }
    } finally {
      setIsStreaming(false);
    }
  }, [options]);

  const applyPolished = useCallback((text: string) => {
    const data = { optimized_resume: text };
    polishedRef.current = data;
    setPolished(data);
    if (options.onPolishedUpdate) {
      options.onPolishedUpdate(data);
    }
  }, [options]);

  const clearOutput = useCallback(() => {
    setScore(null);
    setSuggestions(null);
    setPolished(null);
    setError(null);
    scoreRef.current = null;
    suggestionsRef.current = null;
    polishedRef.current = null;
  }, []);

  return {
    score,
    suggestions,
    polished,
    error,
    isStreaming,
    startStream,
    applyPolished,
    clearOutput
  };
}

export default useStreamResponse;
