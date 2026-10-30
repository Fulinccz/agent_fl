import { useCallback, useRef } from 'react';

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

