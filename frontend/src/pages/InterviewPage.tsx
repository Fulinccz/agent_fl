import React from 'react';
import type { UseStreamResponseReturn } from '../hooks/useStreamResponse';

interface InterviewPageProps {
  stream: UseStreamResponseReturn;
}

interface StrategyGroupProps {
  label: string;
  tone: 'good' | 'risk' | 'tip';
  items?: string[];
}

/** 攻略分组：亮点 / 风险 / 建议，同一套列表样式用 tone 区分标记 */
const StrategyGroup: React.FC<StrategyGroupProps> = ({ label, tone, items }) => {
  if (!items || items.length === 0) return null;
  return (
    <div className={`strategy-group strategy-${tone}`}>
      <span className="strategy-label">{label}</span>
      <ul className="strategy-list">
        {items.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    </div>
  );
};

/** 面试攻略页：整体分析攻略 + 最可能被问的问题 */
const InterviewPage: React.FC<InterviewPageProps> = ({ stream }) => {
  const { interview, isStreaming, elapsed, liveSeconds } = stream;

  const strategy = interview?.strategy;
  const questions = interview?.questions ?? [];
  const hasStrategy = !!(strategy?.summary || strategy?.highlights?.length
    || strategy?.risks?.length || strategy?.tips?.length);

  return (
    <div className="page-stack">
      <div className="card interview-card">
        <div className="card-head">
          <span className="card-title">面试攻略</span>
          {interview?.position && (
            <span className="interview-position">{interview.position}</span>
          )}
          {isStreaming && (
            <span className="streaming-hint">
              {questions.length > 0
                ? `已出 ${questions.length} 个高频问题 · ${liveSeconds}s`
                : `正在生成 · ${liveSeconds}s`}
              <span className="dot-pulse" />
            </span>
          )}
          {elapsed != null && !isStreaming && (
            <span className="pipeline-elapsed">{elapsed}s</span>
          )}
        </div>
        <div className="card-body">
          {hasStrategy && (
            <section className="interview-strategy">
              <div className="strategy-head">
                <span className="strategy-title">整体分析攻略</span>
                {interview?.source === 'rule' && (
                  <span className="strategy-source">含规则兜底</span>
                )}
              </div>
              {strategy?.summary && (
                <p className="strategy-summary">{strategy.summary}</p>
              )}
              <div className="strategy-groups">
                <StrategyGroup label="核心亮点" tone="good" items={strategy?.highlights} />
                <StrategyGroup label="主要风险" tone="risk" items={strategy?.risks} />
                <StrategyGroup label="备战建议" tone="tip" items={strategy?.tips} />
              </div>
            </section>
          )}

          {questions.length > 0 ? (
            <section className="likely-questions">
              <div className="likely-head">
                最可能被问的 {questions.length} 个问题
              </div>
              <ol className="interview-questions">
                {questions.map((q, index) => (
                  <li key={index} className="interview-question">
                    <div className="iq-head">
                      <span className="iq-index">{String(index + 1).padStart(2, '0')}</span>
                      {q.category && <span className="iq-category">{q.category}</span>}
                    </div>
                    <p className="iq-text">{q.question}</p>
                    {q.assess_point && (
                      <p className="iq-meta">
                        <span className="iq-meta-label">考察点</span>
                        {q.assess_point}
                      </p>
                    )}
                    {q.answer_points && q.answer_points.length > 0 && (
                      <div className="iq-points">
                        <span className="iq-meta-label">答题要点</span>
                        <ul>
                          {q.answer_points.map((point, pointIndex) => (
                            <li key={pointIndex}>{point}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {q.follow_up && (
                      <p className="iq-meta iq-follow">
                        <span className="iq-meta-label">可能追问</span>
                        {q.follow_up}
                      </p>
                    )}
                  </li>
                ))}
              </ol>
            </section>
          ) : (
            !hasStrategy && (
              <div className="empty-state">
                <span className="empty-icon">💼</span>
                <p className="empty-title">
                  {isStreaming ? '正在分析简历、准备面试攻略…' : '面试攻略将在这里生成'}
                </p>
                <p className="empty-sub">
                  {isStreaming
                    ? '本地模型并行生成整体攻略与高频问题，通常需要 1-2 分钟'
                    : '输入简历（可选填 JD），点击「生成攻略」'}
                </p>
              </div>
            )
          )}
          {isStreaming && hasStrategy && questions.length === 0 && (
            <p className="empty-sub">正在整理最可能被问的问题…</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default InterviewPage;
