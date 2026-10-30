import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import type { PageKey } from '../types';
import { useFileUpload } from '../hooks/useFileUpload';
import { useVoiceRecorder } from '../hooks/useVoiceRecorder';

/**
 * 三页共用的输入档案
 *
 * 简历文本 / JD / 上传文件 / 语音转写结果只保留一份：
 * 在任一页面填写后切页依然有效，提交时再由各页各自发起请求。
 * 结果不放在这里——结果按页面分桶，避免切页互相清空。
 */
export interface ProfileContextValue {
  query: string;
  jd: string;
  provider: string;
  uploadedFile: File | null;
  voiceState: 'idle' | 'recording' | 'transcribing';
  voiceSeconds: number;
  voiceSupported: boolean;
  voiceError: string;
  /** 当前简历文本是否由上传解析自动回填（用户手动编辑后为 false） */
  queryAutoFilled: boolean;
  setQuery: (value: string) => void;
  setJd: (value: string) => void;
  setProvider: (value: string) => void;
  setVoiceError: (value: string) => void;
  handleFileSelect: (event: ChangeEvent<HTMLInputElement>) => void;
  clearFile: () => void;
  toggleVoice: () => void;
  /** 上传解析结果回填简历输入框（用户已填补充要求时不覆盖） */
  backfillResume: (text: string) => void;
  /** 新对话：清空输入（不动各页已产出的结果） */
  resetInputs: () => void;
}

export const ProfileContext = createContext<ProfileContextValue | null>(null);

export function useProfile(): ProfileContextValue {
  const value = useContext(ProfileContext);
  if (!value) {
    throw new Error('useProfile 必须在 ProfileProvider 内使用');
  }
  return value;
}

/** 页面入口 → 输入区文案与提交按钮文案 */
export const PAGE_COPY: Record<PageKey, {
  resumePlaceholder: string;
  submitLabel: string;
  jdHint: string;
}> = {
  analyze: {
    resumePlaceholder: '粘贴简历内容，或点击左下角上传文档 / 图片 / 音频…',
    submitLabel: '开始分析 ↑',
    jdHint: '填写后可额外得到 JD 匹配分析'
  },
  generate: {
    resumePlaceholder: '粘贴简历内容，AI 将生成可直接投递的成品简历…',
    submitLabel: '生成简历 ↑',
    jdHint: '填写后成品会向目标岗位靠拢'
  },
  interview: {
    resumePlaceholder: '粘贴简历内容，AI 将按维度生成面试题库…',
    submitLabel: '生成攻略 ↑',
    jdHint: '填写后题目会围绕目标岗位展开'
  }
};

/** 供 ProfileProvider 组装 context 值（hooks 必须在组件内调用） */
export function useProfileState(): ProfileContextValue {
  const [query, setQueryState] = useState('');
  const [queryAutoFilled, setQueryAutoFilled] = useState(false);
  const [jd, setJd] = useState('');
  const [provider, setProvider] = useState('deepseek');
  const [voiceError, setVoiceError] = useState('');

  // ref 只用于事件回调读取最新输入，避免把过期闭包里的 query 写回去
  const queryRef = useRef('');
  useEffect(() => {
    queryRef.current = query;
  }, [query]);

  // 任何用户输入（打字 / 语音）都视为手动内容，不再当作"自动回填"
  const setQuery = useCallback((value: string) => {
    setQueryAutoFilled(false);
    setQueryState(value);
  }, []);

  const { uploadedFile, handleFileSelect, clearFile } = useFileUpload();

  // 语音输入：录音 → 后端本地 Whisper 转写 → 追加到简历输入框（不覆盖已有内容）
  const {
    state: voiceState,
    seconds: voiceSeconds,
    supported: voiceSupported,
    toggle: toggleVoice
  } = useVoiceRecorder({
    onText: (text) => {
      setVoiceError('');
      setQuery(queryRef.current.trim() ? `${queryRef.current.trimEnd()}\n${text}` : text);
    },
    onError: (message) => setVoiceError(message)
  });

  const backfillResume = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    // 用户已经写了补充要求时不覆盖；仅在输入框为空时回填并标记来源
    if (queryRef.current.trim()) return;
    setQueryState(trimmed);
    setQueryAutoFilled(true);
  }, []);

  const resetInputs = useCallback(() => {
    setQueryState('');
    setQueryAutoFilled(false);
    setJd('');
    setVoiceError('');
    clearFile();
  }, [clearFile]);

  return useMemo<ProfileContextValue>(() => ({
    query,
    jd,
    provider,
    uploadedFile,
    voiceState,
    voiceSeconds,
    voiceSupported,
    voiceError,
    queryAutoFilled,
    setQuery,
    setJd,
    setProvider,
    setVoiceError,
    handleFileSelect,
    clearFile,
    toggleVoice,
    backfillResume,
    resetInputs
  }), [
    query, jd, provider, uploadedFile, voiceState, voiceSeconds,
    voiceSupported, voiceError, queryAutoFilled, setQuery, handleFileSelect, clearFile,
    toggleVoice, backfillResume, resetInputs
  ]);
}
