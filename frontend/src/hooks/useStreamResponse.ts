import { useState, useCallback, useRef, useEffect } from 'react';
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
  InterviewData,
  InterviewGuideData,
  ParsedResumeData,
} from '../types';

interface UseStreamResponseOptions {
  onScoreUpdate?: (score: ScoreData) => void;
  onSuggestionsUpdate?: (suggestions: SuggestionsData) => void;
  onPolishedUpdate?: (polished: PolishedData) => void;
  onStreamError?: (error: string) => void;
  /** 上传链路解析出的简历正文：回填三页共用的输入框 */
  onResumeParsed?: (text: string) => void;
}

export interface UseStreamResponseReturn {
  score: ScoreData | null;
  suggestions: SuggestionsData | null;
  polished: PolishedData | null;
  condensed: CondensedData | null;
  /** 面试攻略（分维度题库，结构化卡片） */
  interview: InterviewGuideData | null;
  error: string | null;
  isStreaming: boolean;
  /** 本次流式已用秒数（生成期间每秒刷新，结束后归零） */
  liveSeconds: number;
  elapsed: number | null;
  /** ReAct 步内思考过程（thought 事件按序累积） */
  thoughts: string[];
  startStream: (request: ResumeOptimizeRequest, signal?: AbortSignal) => Promise<void>;
  startUpload: (request: UploadRequest, signal?: AbortSignal) => Promise<void>;
  clearOutput: () => void;
  /** 手动关闭错误提示（不触发重新请求） */
  clearError: () => void;
}

export function useStreamResponse(
  options: UseStreamResponseOptions = {}
): UseStreamResponseReturn {
  const [score, setScore] = useState<ScoreData | null>(null);
  const [suggestions, setSuggestions] = useState<SuggestionsData | null>(null);
  const [polished, setPolished] = useState<PolishedData | null>(null);
  const [condensed, setCondensed] = useState<CondensedData | null>(null);
  const [interview, setInterview] = useState<InterviewGuideData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [liveSeconds, setLiveSeconds] = useState(0);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [thoughts, setThoughts] = useState<string[]>([]);
  const thoughtsRef = useRef<string[]>([]);

  const scoreRef = useRef<ScoreData | null>(null);
  const suggestionsRef = useRef<SuggestionsData | null>(null);
  const polishedRef = useRef<PolishedData | null>(null);
  const condensedRef = useRef<CondensedData | null>(null);
  const interviewRef = useRef<InterviewGuideData | null>(null);
  const lastPolishTs = useRef(0);

  // options 每次渲染可能是新字面量，用 ref 持有最新值，
  // 使下方回调 identity 稳定（避免依赖 [options] 导致每渲染重建）
  const optionsRef = useRef(options);
  optionsRef.current = options;

  // 生成期间每秒刷新已用秒数：本地模型一次推理 1-3 分钟，
  // 有计时器用户才不会误以为页面卡死。归零在 runStream 里做（避免 effect 内同步 setState）。
  useEffect(() => {
    if (!isStreaming) return;
    const startAt = Date.now();
    const timer = window.setInterval(() => {
      setLiveSeconds(Math.floor((Date.now() - startAt) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [isStreaming]);

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
      case 'interview': {
        const data = (event.data || {}) as InterviewData;
        const guide = data.interview_guide || null;
        if (guide) {
          interviewRef.current = guide;
          setInterview(guide);
        }
        break;
      }
      case 'resume_parsed': {
        const data = (event.data || {}) as ParsedResumeData;
        const text = (data.text || '').trim();
        if (text) opts.onResumeParsed?.(text);
        break;
      }
      case 'complete': {
        const data = event.data as CompleteData | undefined;
        if (data?.elapsed_seconds != null) {
          setElapsed(data.elapsed_seconds);
        }
        break;
      }
      case 'thought': {
        // ReAct 步内思考：按序累积，重复内容跳过
        const content = (event.content || '').trim();
        if (content && thoughtsRef.current[thoughtsRef.current.length - 1] !== content) {
          const next = [...thoughtsRef.current, content];
          thoughtsRef.current = next;
          setThoughts(next);
        }
        break;
      }
      case 'error': {
        const msg = event.message || '服务器处理出错';
        setError(msg);
        opts.onStreamError?.(msg);
        break;
      }
      default:
        // intent / plan / step / reflection 等过程事件：当前 UI 不消费
        break;
    }
  }, []);

  const resetOutput = useCallback(() => {
    setScore(null);
    setSuggestions(null);
    setPolished(null);
    setCondensed(null);
    setInterview(null);
    setError(null);
    setElapsed(null);
    setThoughts([]);
    scoreRef.current = null;
    suggestionsRef.current = null;
    polishedRef.current = null;
    condensedRef.current = null;
    interviewRef.current = null;
    thoughtsRef.current = [];
  }, []);

  const runStream = useCallback(async (
    starter: (onEvent: (event: ResumeOptimizeEvent) => void) => Promise<void>
  ) => {
    resetOutput();
    setLiveSeconds(0);
    setIsStreaming(true);
    try {
      await starter(handleEvent);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '请求失败';
      setError(msg);
      optionsRef.current.onStreamError?.(msg);
    } finally {
      setIsStreaming(false);
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

  const clearError = useCallback(() => setError(null), []);

  return {
    score,
    suggestions,
    polished,
    condensed,
    interview,
    error,
    isStreaming,
    liveSeconds,
    elapsed,
    thoughts,
    startStream,
    startUpload,
    clearOutput,
    clearError,
  };
}

export default useStreamResponse;
