import { useCallback, useEffect, useRef, useState } from 'react';
import { uploadApiClient } from '../services/api';

export type VoiceState = 'idle' | 'recording' | 'transcribing';

/** 录音时长上限：超过后自动停止并转写（控制上传体积与转写等待） */
const MAX_SECONDS = 60;

interface UseVoiceRecorderOptions {
  /** 转写成功：把文本交给调用方写入输入框 */
  onText: (text: string) => void;
  /** 失败：统一交给调用方展示（麦克风权限/转写错误） */
  onError: (message: string) => void;
}

interface UseVoiceRecorderReturn {
  state: VoiceState;
  seconds: number;
  supported: boolean;
  toggle: () => void;
}

/** MediaRecorder 输出容器 → 扩展名（后端按扩展名分流 STT，webm 需魔数校验通过） */
function pickContainer(mimeType: string): { filename: string; type: string } {
  const mime = mimeType || 'audio/webm';
  // Safari 输出 audio/mp4，后端 .m4a 与 ftyp 魔数可识别
  if (mime.includes('mp4')) return { filename: 'voice.m4a', type: 'audio/mp4' };
  if (mime.includes('ogg')) return { filename: 'voice.ogg', type: 'audio/ogg' };
  return { filename: 'voice.webm', type: 'audio/webm' };
}

function micErrorMessage(err: unknown): string {
  const name = (err as { name?: string })?.name ?? '';
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return '麦克风权限被拒绝，请在浏览器地址栏允许麦克风后重试';
    case 'NotFoundError':
      return '未检测到麦克风设备';
    case 'NotReadableError':
      return '麦克风被其他应用占用，请关闭后重试';
    default: {
      const message = (err as { message?: string })?.message;
      return message ? `麦克风启动失败：${message}` : '麦克风启动失败';
    }
  }
}

/**
 * 浏览器录音 → 后端本地 Whisper 转写 → 返回文本。
 *
 * 音频不出本机（私有化部署前提），转写结果由用户校对后再触发生成。
 * 需要安全上下文：localhost 或 HTTPS，内网 http + IP 时浏览器会拒绝授权。
 */
export function useVoiceRecorder({
  onText,
  onError
}: UseVoiceRecorderOptions): UseVoiceRecorderReturn {
  const [state, setState] = useState<VoiceState>('idle');
  const [seconds, setSeconds] = useState(0);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  // 回调放 ref：录音期间父组件重渲染不会让 onstop 里的闭包失效
  const onTextRef = useRef(onText);
  const onErrorRef = useRef(onError);
  useEffect(() => {
    onTextRef.current = onText;
    onErrorRef.current = onError;
  }, [onText, onError]);

  const supported =
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== 'undefined';

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // 必须显式 stop track，否则标签页一直显示"正在录音"
  const releaseStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  const start = useCallback(async () => {
    if (state !== 'idle') return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const recorder = new MediaRecorder(
        stream,
        MediaRecorder.isTypeSupported('audio/webm') ? { mimeType: 'audio/webm' } : undefined
      );
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };

      recorder.onstop = () => {
        clearTimer();
        releaseStream();
        recorderRef.current = null;

        const chunks = chunksRef.current;
        chunksRef.current = [];
        const { filename, type } = pickContainer(recorder.mimeType);
        const blob = new Blob(chunks, { type });

        if (blob.size === 0) {
          setState('idle');
          onErrorRef.current('没有录到音频，请检查麦克风后重试');
          return;
        }

        setState('transcribing');
        uploadApiClient
          .transcribe(new File([blob], filename, { type }))
          .then((text) => {
            const trimmed = text.trim();
            if (trimmed) onTextRef.current(trimmed);
            else onErrorRef.current('没有识别到语音内容，请靠近麦克风、放慢语速后重试');
          })
          .catch((err: unknown) => {
            onErrorRef.current(err instanceof Error ? err.message : '语音转写失败');
          })
          .finally(() => setState('idle'));
      };

      recorder.start();
      setSeconds(0);
      setState('recording');
      timerRef.current = window.setInterval(() => setSeconds((prev) => prev + 1), 1000);
    } catch (err) {
      releaseStream();
      setState('idle');
      onErrorRef.current(micErrorMessage(err));
    }
  }, [state, clearTimer, releaseStream]);

  const toggle = useCallback(() => {
    if (state === 'recording') {
      recorderRef.current?.stop();
      return;
    }
    if (state === 'idle') void start();
  }, [state, start]);

  // 到达时长上限自动停止（副作用不放进 setState 更新函数，避免 StrictMode 双调用）
  useEffect(() => {
    if (state === 'recording' && seconds >= MAX_SECONDS) {
      recorderRef.current?.stop();
    }
  }, [state, seconds]);

  // 卸载时释放麦克风
  useEffect(
    () => () => {
      clearTimer();
      recorderRef.current?.stop();
      releaseStream();
    },
    [clearTimer, releaseStream]
  );

  return { state, seconds, supported, toggle };
}

export default useVoiceRecorder;
