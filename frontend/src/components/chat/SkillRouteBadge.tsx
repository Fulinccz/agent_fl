import React from 'react';
import type { SkillRouteState } from '../../types/skill';

interface SkillRouteBadgeProps {
  routeState: SkillRouteState;
}

/**
 * 技能路由状态徽章组件
 * 
 * 展示当前意图识别的结果：
 * - 识别中的加载状态
 * - 识别成功：显示技能名称、置信度和来源
 * - 识别失败：显示错误信息
 */
const SkillRouteBadge: React.FC<SkillRouteBadgeProps> = ({ routeState }) => {
  const { currentSkill, confidence, source, reason, isRouting, error } = routeState;

  // 获取来源对应的样式
  const getSourceStyle = (source: string) => {
    switch (source) {
      case 'keyword':
        return { bg: '#e8f5e9', color: '#2e7d32', label: '关键词' };
      case 'vector':
        return { bg: '#e3f2fd', color: '#1565c0', label: '向量' };
      case 'llm':
        return { bg: '#fff3e0', color: '#e65100', label: '模型' };
      default:
        return { bg: '#f5f5f5', color: '#616161', label: source || '未知' };
    }
  };

  const sourceStyle = getSourceStyle(source);

  // 获取技能中文名称
  const getSkillLabel = (skill: string) => {
    const skillLabels: Record<string, string> = {
      'resume-polishing': '简历润色',
      'jd-keyword-match': '岗位匹配',
      'resume-score': '简历评分',
      'resume-parse': '简历解析',
      'chat-general': '通用对话'
    };
    return skillLabels[skill] || skill;
  };

  if (isRouting) {
    return (
      <div className="skill-route-badge routing">
        <span className="route-spinner">⟳</span>
        <span className="route-text">正在识别意图...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="skill-route-badge error">
        <span className="route-icon">⚠</span>
        <span className="route-text">识别失败: {error}</span>
      </div>
    );
  }

  if (!currentSkill) {
    return null;
  }

  return (
    <div className="skill-route-badge">
      <span className="route-label">识别到技能:</span>
      <span className="route-skill">{getSkillLabel(currentSkill)}</span>
      <span
        className="route-source"
        style={{
          backgroundColor: sourceStyle.bg,
          color: sourceStyle.color
        }}
      >
        {sourceStyle.label}
      </span>
      <span className="route-confidence">
        置信度: {(confidence * 100).toFixed(0)}%
      </span>
      {reason && (
        <span className="route-reason" title={reason}>
          {reason.length > 20 ? reason.slice(0, 20) + '...' : reason}
        </span>
      )}
    </div>
  );
};

export default SkillRouteBadge;
