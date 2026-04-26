import { useState, useCallback, useRef } from 'react';
import { agentService } from '../services/agentService';
import type { ResumeOptimizeRequest } from '../types';

interface UseStreamResponseOptions {
  onScoreUpdate?: (score: { overall_score: any; scores: any }) => void;
  onSuggestionsUpdate?: (suggestions: { suggestions: any; match_analysis: any }) => void;
  onPolishedUpdate?: (polished: { optimized_resume: string }) => void;
}

interface UseStreamResponseReturn {
  score: { overall_score: any; scores: any } | null;
  suggestions: { suggestions: any; match_analysis: any } | null;
  polished: { optimized_resume: string } | null;
  isStreaming: boolean;
  startStream: (request: ResumeOptimizeRequest, signal?: AbortSignal) => Promise<void>;
  clearOutput: () => void;
}

export function useStreamResponse(
  options: UseStreamResponseOptions = {}
): UseStreamResponseReturn {
  const [score, setScore] = useState<{ overall_score: any; scores: any } | null>(null);
  const [suggestions, setSuggestions] = useState<{ suggestions: any; match_analysis: any } | null>(null);
  const [polished, setPolished] = useState<{ optimized_resume: string } | null>(null);
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
          polishedRef.current = data;
          setPolished(data);
          if (options.onPolishedUpdate) {
            options.onPolishedUpdate(data);
          }
        }
      );
    } finally {
      setIsStreaming(false);
    }
  }, [options]);

  const clearOutput = useCallback(() => {
    setScore(null);
    setSuggestions(null);
    setPolished(null);
    scoreRef.current = null;
    suggestionsRef.current = null;
    polishedRef.current = null;
  }, []);

  return {
    score,
    suggestions,
    polished,
    isStreaming,
    startStream,
    clearOutput
  };
}

export default useStreamResponse;
