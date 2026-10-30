import React from 'react';
import { ChatInput } from '../components/chat';
import type { PageKey } from '../types';
import { PAGE_COPY, useProfile } from './profile-context';

interface ProfileFormProps {
  page: PageKey;
  /** 当前页面是否正在流式生成 */
  isStreaming: boolean;
  /** 其他页面正在生成：全局只允许一条流，避免本地推理队列堵塞 */
  blockedByOther: boolean;
  onSubmit: () => void;
  onStop: () => void;
}

/**
 * 三页共用的输入卡：简历文本 / JD / 附件 / 语音
 *
 * 输入来自 ProfileContext，切页不丢失；提交动作由所在页面决定（携带 page 参数）。
 */
const ProfileForm: React.FC<ProfileFormProps> = ({
  page,
  isStreaming,
  blockedByOther,
  onSubmit,
  onStop
}) => {
  const profile = useProfile();
  const copy = PAGE_COPY[page];

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    onSubmit();
  };

  return (
    <form className="card input-card" onSubmit={handleSubmit}>
      <textarea
        className="resume-input"
        placeholder={copy.resumePlaceholder}
        value={profile.query}
        onChange={(e) => profile.setQuery(e.target.value)}
        disabled={isStreaming}
        rows={7}
      />
      <div className="jd-block">
        <label className="jd-label">
          目标岗位 JD（可选）
          <span className="jd-hint">{copy.jdHint}</span>
        </label>
        <textarea
          className="jd-input"
          placeholder="粘贴职位描述，匹配分析 / 出题方向将围绕它展开"
          value={profile.jd}
          onChange={(e) => profile.setJd(e.target.value)}
          disabled={isStreaming}
          rows={4}
        />
      </div>
      <ChatInput
        onSubmit={onSubmit}
        onStop={onStop}
        isLoading={isStreaming}
        uploadedFile={profile.uploadedFile}
        onFileChange={profile.handleFileSelect}
        onFileClear={profile.clearFile}
        disabled={blockedByOther}
        isRecording={profile.voiceState === 'recording'}
        isTranscribing={profile.voiceState === 'transcribing'}
        recordSeconds={profile.voiceSeconds}
        voiceSupported={profile.voiceSupported}
        onVoiceToggle={profile.toggleVoice}
        submitLabel={copy.submitLabel}
        hasInput={profile.query.trim() !== '' || !!profile.uploadedFile}
      />
    </form>
  );
};

export default ProfileForm;
