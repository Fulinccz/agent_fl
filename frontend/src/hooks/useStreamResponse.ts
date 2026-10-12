import { useState, useCallback, useRef } from 'react';
import { agentService } from '../services/agentService';
import type {
  ResumeOptimizeRequest,
  UploadRequest,
  ResumeOptimizeEvent,
  ScoreData,
  SuggestionsData,
  PolishedData,
  CondensedData,
  CompleteData,
  StepStartData,
  StepResultData,
} from '../types';

/** agent 名称 → pipeline 步骤 key */
export const AGENT_TO_PIPELINE: Record<string, string> = {
  resume_score: 'score',
  jd_match: 'suggestions',
  resume_condense: 'condense',
  resume_polish: 'polished',
};

interface UseStreamResponseOptions {
  onScoreUpdate?: (score: ScoreData) => void;
  onSuggestionsUpdate?: (suggestions: SuggestionsData) => void;
  onPolishedUpdate?: (polished: PolishedData) => void;
  onStreamError?: (error: string) => void;
}

interface UseStreamResponseReturn {
  score: ScoreData | null;
  suggestions: SuggestionsData | null;
  polished: PolishedData | null;
  condensed: CondensedData | null;
  error: string | null;
  isStreaming: boolean;
  activeAgents: Set<string>;
  elapsed: number | null;
  startStream: (request: ResumeOptimizeRequest, signal?: AbortSignal) => Promise<void>;
  startUpload: (request: UploadRequest, signal?: AbortSignal) => Promise<void>;
  clearOutput: () => void;
}

export function useStreamResponse(
  options: UseStreamResponseOptions = {}
): UseStreamResponseReturn {
  const [score, setScore] = useState<ScoreData | null>(null);
  const [suggestions, setSuggestions] = useState<SuggestionsData | null>(null);
  const [polished, setPolished] = useState<PolishedData | null>(null);
  const [condensed, setCondensed] = useState<CondensedData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeAgents, setActiveAgents] = useState<Set<string>>(new Set());
  const [elapsed, setElapsed] = useState<number | null>(null);

  const scoreRef = useRef<ScoreData | null>(null);
  const suggestionsRef = useRef<SuggestionsData | null>(null);
  const polishedRef = useRef<PolishedData | null>(null);
  const condensedRef = useRef<CondensedData | null>(null);
  const lastPolishTs = useRef(0);
  const activeAgentsRef = useRef<Set<string>>(new Set());

  // options 每次渲染可能是新字面量，用 ref 持有最新值，
  // 使下方回调 identity 稳定（避免依赖 [options] 导致每渲染重建）
  const optionsRef = useRef(options);
  optionsRef.current = options;

  /** 统一事件分发：粘贴 / 上传两条链路共用 */
  const handleEvent = useCallback((event: ResumeOptimizeEvent) => {
    const opts = optionsRef.current;
    switch (event.type) {
      case 'score': {
        const data = event.data as ScoreData;
        scoreRef.current = data;
        setScore(data);
        opts.onScoreUpdate?.(data);
        break;
      }
      case 'suggestions': {
        const data = event.data as SuggestionsData;
        suggestionsRef.current = data;
        setSuggestions(data);
        opts.onSuggestionsUpdate?.(data);
        break;
      }
      case 'polished': {
        const data = event.data as PolishedData;
        const isPartial = data.partial === true;
        const newText = data.optimized_resume || '';
        const currentText = polishedRef.current?.optimized_resume || '';

        if (isPartial) {
          // 节流：partial 更新至少间隔 120ms，减少 React 重渲染
          const now = Date.now();
          if (newText !== currentText && now - lastPolishTs.current > 120) {
            lastPolishTs.current = now;
            polishedRef.current = data;
            setPolished(data);
            opts.onPolishedUpdate?.(data);
          }
        } else if (newText !== currentText) {
          // 最终结果：总是更新
          polishedRef.current = data;
          setPolished(data);
          opts.onPolishedUpdate?.(data);
        }
        break;
      }
      case 'condensed': {
        const data = event.data as CondensedData;
        condensedRef.current = data;
        setCondensed(data);
        break;
      }
      case 'complete': {
        const data = event.data as CompleteData | undefined;
        if (data?.elapsed_seconds != null) {
          setElapsed(data.elapsed_seconds);
        }
        break;
      }
      case 'error': {
        const msg = event.message || '服务器处理出错';
        setError(msg);
        opts.onStreamError?.(msg);
        break;
      }
      case 'step_start': {
        const data = event.data as StepStartData;
        if (data.agent) {
          const next = new Set([...activeAgentsRef.current, data.agent]);
          activeAgentsRef.current = next;
          setActiveAgents(next);
        }
        break;
      }
      case 'step_result': {
        const data = event.data as StepResultData;
        if (data.agent) {
          const next = new Set(activeAgentsRef.current);
          next.delete(data.agent);
          activeAgentsRef.current = next;
          setActiveAgents(next);
        }
        break;
      }
      default:
        // intent / plan / reflection 等过程事件：当前 UI 不消费
        break;
    }
  }, []);

  const resetOutput = useCallback(() => {
    setScore(null);
    setSuggestions(null);
    setPolished(null);
    setCondensed(null);
    setError(null);
    setElapsed(null);
    setActiveAgents(new Set());
    scoreRef.current = null;
    suggestionsRef.current = null;
    polishedRef.current = null;
    condensedRef.current = null;
    activeAgentsRef.current = new Set();
  }, []);

  const runStream = useCallback(async (
    starter: (onEvent: (event: ResumeOptimizeEvent) => void) => Promise<void>
  ) => {
    resetOutput();
    setIsStreaming(true);
    try {
      await starter(handleEvent);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '请求失败';
      setError(msg);
      optionsRef.current.onStreamError?.(msg);
    } finally {
      setIsStreaming(false);
      setActiveAgents(new Set());
      activeAgentsRef.current = new Set();
    }
  }, [handleEvent, resetOutput]);

  const startStream = useCallback((
    request: ResumeOptimizeRequest,
    signal?: AbortSignal
  ) => {
    return runStream((onEvent) => agentService.runOptimizeStream(request, signal, onEvent));
  }, [runStream]);

  const startUpload = useCallback((
    request: UploadRequest,
    signal?: AbortSignal
  ) => {
    return runStream((onEvent) => agentService.runUploadStream(request, signal, onEvent));
  }, [runStream]);

  const clearOutput = useCallback(() => {
    resetOutput();
    setIsStreaming(false);
  }, [resetOutput]);

  return {
    score,
    suggestions,
    polished,
    condensed,
    error,
    isStreaming,
    activeAgents,
    elapsed,
    startStream,
    startUpload,
    clearOutput,
  };
}

export default useStreamResponse;
