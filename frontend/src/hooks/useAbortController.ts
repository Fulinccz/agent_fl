import { useState, useCallback, useRef } from 'react';

interface UseAbortControllerReturn {
  abortController: AbortController | null;
  isAborted: boolean;
  create: () => AbortController;
  abort: () => void;
  reset: () => void;
}

export function useAbortController(): UseAbortControllerReturn {
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  const [isAborted, setIsAborted] = useState(false);
  
  const isSubmittingRef = useRef(false);
  const lastStopTimeRef = useRef(0);

  const create = useCallback((): AbortController => {
    const controller = new AbortController();
    setAbortController(controller);
    setIsAborted(false);
    isSubmittingRef.current = false;
    return controller;
  }, []);

  const abort = useCallback(() => {
    if (abortController && !abortController.signal.aborted) {
      abortController.abort();
      setIsAborted(true);
      lastStopTimeRef.current = Date.now();
      
      console.log('⛔ 用户请求中止生成');
    }
  }, [abortController]);

  const reset = useCallback(() => {
    setAbortController(null);
    setIsAborted(false);
    isSubmittingRef.current = false;
  }, []);

  return {
    abortController,
    isAborted,
    create,
    abort,
    reset
  };
}

const STOP_COOLDOWN_MS = 2000;

export function useSubmitControl() {
  const isSubmittingRef = useRef(false);
  const lastStopTimeRef = useRef(0);

  const canSubmit = useCallback((): boolean => {
    if (isSubmittingRef.current) return false;
    
    const timeSinceLastStop = Date.now() - lastStopTimeRef.current;
    if (timeSinceLastStop < STOP_COOLDOWN_MS) return false;
    
    return true;
  }, []);

  const markSubmitting = useCallback((submitting: boolean) => {
    isSubmittingRef.current = submitting;
  }, []);

  const recordStopTime = useCallback(() => {
    lastStopTimeRef.current = Date.now();
  }, []);

  return {
    canSubmit,
    markSubmitting,
    recordStopTime
  };
}

export default useAbortController;
