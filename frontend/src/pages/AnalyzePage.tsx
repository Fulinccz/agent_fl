import React from 'react';
import type { UseStreamResponseReturn } from '../hooks/useStreamResponse';
import type { MatchAnalysis, SuggestionItem } from '../types';

interface AnalyzePageProps {
  stream: UseStreamResponseReturn;
  /** 本轮提交是否带了 JD（决定是否出现匹配卡） */
  hasJd: boolean;
}

/** 评分档位 → 配色（与后端 _get_rating 的档位名保持一致） */
const RATING_STYLES: Record<string, { color: string; bg: string }> = {
  '优秀': { color: '#16a34a', bg: 'rgba(22, 163, 74, 0.12)' },
  '良好': { color: '#2563eb', bg: 'rgba(37, 99, 235, 0.12)' },
  '中等': { color: '#d97706', bg: 'rgba(217, 119, 6, 0.12)' },
  '及格': { color: '#ea580c', bg: 'rgba(234, 88, 12, 0.12)' },
  '需改进': { color: '#dc2626', bg: 'rgba(220, 38, 38, 0.12)' },
};

const DIMENSION_LABELS: Record<string, string> = {
  completeness: '完整性',
  professionalism: '专业度',
  quantification: '量化程度',
  matching: '匹配度',
};

function scoreColor(value: number) {
  return value >= 80 ? '#16a34a' : value >= 60 ? '#d97706' : '#dc2626';
}

function matchColor(value: number) {
  return value >= 80 ? '#16a34a' : value >= 60 ? '#d97706' : '#dc2626';
}

/** 简历分析页：总分 + 维度分析 + 改进建议 + JD 匹配（有 JD 才有） */
const AnalyzePage: React.FC<AnalyzePageProps> = ({ stream, hasJd }) => {
  const { score, suggestions, isStreaming, elapsed, liveSeconds } = stream;

  const scoreValue = score?.overall_score?.score;
  const rating = score?.overall_score?.rating ?? '';
  const ratingStyle = RATING_STYLES[rating]
    ?? { color: '#4f5bd5', bg: 'rgba(79, 91, 213, 0.12)' };
  const dimensionAnalysis = (score?.dimension_analysis ?? {}) as Record<
    string,
    { assessment?: string; suggestion?: string } | string
  >;

  const match = (suggestions?.match_analysis ?? {}) as MatchAnalysis;
  const items: SuggestionItem[] = Array.isArray(suggestions?.suggestions)
    ? (suggestions!.suggestions as SuggestionItem[])
    : [];
  const matchedKeywords = Array.isArray(match.matched_keywords) ? match.matched_keywords : [];
  const missingKeywords = Array.isArray(match.missing_keywords) ? match.missing_keywords : [];
  const matchScore = typeof match.match_score === 'number' ? match.match_score : null;
  const showMatchCard = hasJd || !!suggestions;

  return (
    <div className="page-stack">
      <div className="card score-card">
        <div className="card-head">
          <span className="card-title">简历评分</span>
          {isStreaming && (
            <span className="streaming-hint">
              分析中 · {liveSeconds}s<span className="dot-pulse" />
            </span>
          )}
          {elapsed != null && !isStreaming && (
            <span className="pipeline-elapsed">{elapsed}s</span>
          )}
        </div>
        <div className="card-body score-body">
          {score && scoreValue != null ? (
            <div className="score-wrap">
              <div className="score-left">
                <div
                  className="score-ring"
                  style={{
                    background: `conic-gradient(${ratingStyle.color} ${Math.min(100, Number(scoreValue)) * 3.6}deg, var(--border) 0deg)`
                  }}
                >
                  <div className="score-ring-inner">
                    <span className="score-number">{scoreValue}</span>
                  </div>
                </div>
                <span
                  className="rating-chip"
                  style={{ color: ratingStyle.color, background: ratingStyle.bg }}
                >
                  {rating || '—'}
                </span>
                {score.overall_score?.description && (
                  <p className="score-overall-desc">{score.overall_score.description}</p>
                )}
              </div>
              {score.scores && typeof score.scores === 'object' && (
                <div className="score-dimensions">
                  {Object.entries(score.scores).map(([key, val]) => {
                    const label = DIMENSION_LABELS[key] || key;
                    const value = Number(val) || 0;
                    const color = scoreColor(value);
                    const item = dimensionAnalysis[key];
                    const assessment = typeof item === 'string' ? item : item?.assessment;
                    const suggestion = typeof item === 'string' ? '' : item?.suggestion;
                    return (
                      <div key={key} className="dim-block">
                        <div className="dim-row">
                          <span className="dim-label">{label}</span>
                          <div className="dim-bar-track">
                            <div className="dim-bar-fill" style={{ width: `${value}%`, background: color }} />
                          </div>
                          <span className="dim-value" style={{ color }}>{value}</span>
                        </div>
                        {assessment && <p className="dim-analysis">{assessment}</p>}
                        {suggestion && (
                          <p className="dim-suggestion">
                            <span className="dim-suggestion-icon">💡</span>
                            {suggestion}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ) : (
            <div className="empty-state compact">
              <span className="empty-icon">📊</span>
              <p className="empty-sub">
                {isStreaming ? '正在从四个维度评估简历…' : '发送简历后出分'}
              </p>
            </div>
          )}
        </div>
      </div>

      {showMatchCard && (
        <div className="card match-card">
          <div className="card-head">
            <span className="card-title">JD 匹配与改进建议</span>
            {matchScore != null && (
              <span className="match-score" style={{ color: matchColor(matchScore) }}>
                匹配度 {matchScore}
              </span>
            )}
          </div>
          <div className="card-body">
            {suggestions ? (
              <>
                {(matchedKeywords.length > 0 || missingKeywords.length > 0) && (
                  <div className="keyword-groups">
                    {matchedKeywords.length > 0 && (
                      <div className="keyword-group">
                        <span className="keyword-label">已覆盖</span>
                        {matchedKeywords.map((kw, i) => (
                          <span key={`m-${i}`} className="keyword-chip">{kw}</span>
                        ))}
                      </div>
                    )}
                    {missingKeywords.length > 0 && (
                      <div className="keyword-group">
                        <span className="keyword-label">待补强</span>
                        {missingKeywords.map((kw, i) => (
                          <span key={`x-${i}`} className="keyword-chip missing">{kw}</span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                {items.length > 0 ? (
                  <ol className="suggestion-list">
                    {items.map((item, index) => (
                      <li key={index} className="suggestion-item">
                        {item.category && (
                          <span className="suggestion-cat">{item.category}</span>
                        )}
                        <p className="suggestion-text">{item.suggestion}</p>
                        {item.example && (
                          <p className="suggestion-example">改写示例：{item.example}</p>
                        )}
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="empty-sub">暂无改进建议</p>
                )}
              </>
            ) : (
              <div className="empty-state compact">
                <span className="empty-icon">🎯</span>
                <p className="empty-sub">
                  {isStreaming ? '正在对照 JD 分析差距…' : '发送简历后出匹配分析'}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalyzePage;
