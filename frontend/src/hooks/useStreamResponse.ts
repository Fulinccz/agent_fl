import { useState, useCallback, useRef } from 'react';
import { agentService } from '../services/agentService';

interface UseStreamResponseOptions {
  onThoughtUpdate?: (thoughts: string[]) => void;
  onResponseUpdate?: (response: string) => void;
}

interface UseStreamResponseReturn {
  thoughts: string[];
  response: string;
  isStreaming: boolean;
  startStream: (query: string, signal?: AbortSignal, deepThinking?: boolean) => Promise<void>;
  clearOutput: () => void;
}

export function useStreamResponse(
  options: UseStreamResponseOptions = {}
): UseStreamResponseReturn {
  const [thoughts, setThoughts] = useState<string[]>([]);
  const [response, setResponse] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  
  const responseRef = useRef('');
  const thoughtsRef = useRef<string[]>([]);

  const startStream = useCallback(async (
    query: string,
    signal?: AbortSignal,
    deepThinking: boolean = false
  ) => {
    setThoughts([]);
    setResponse('');
    responseRef.current = '';
    thoughtsRef.current = [];
    setIsStreaming(true);

    try {
      await agentService.streamQuery(
        { query, deepThinking },
        signal,
        (content) => {
          thoughtsRef.current.push(content);
          setThoughts([...thoughtsRef.current]);
          if (options.onThoughtUpdate) {
            options.onThoughtUpdate(thoughtsRef.current);
          }
        },
        (content) => {
          responseRef.current += content;
          setResponse(responseRef.current);
          if (options.onResponseUpdate) {
            options.onResponseUpdate(responseRef.current);
          }
        }
      );
    } finally {
      setIsStreaming(false);
    }
  }, [options]);

  const clearOutput = useCallback(() => {
    setThoughts([]);
    setResponse('');
    responseRef.current = '';
    thoughtsRef.current = [];
  }, []);

  return {
    thoughts,
    response,
    isStreaming,
    startStream,
    clearOutput
  };
}

export default useStreamResponse;
