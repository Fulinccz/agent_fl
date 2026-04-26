export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

export interface UploadRequest {
  file: File;
  query: string;
  provider?: string;
}

// ==================== 简历优化多 Agent 接口 ====================

export interface ResumeOptimizeRequest {
  resume: string;
  jd?: string;
  position_type?: string;
}

export interface ResumeOptimizeResponse {
  success: boolean;
  overall_score?: {
    score: number;
    rating: string;
    description: string;
  };
  scores?: {
    completeness: number;
    professionalism: number;
    quantification: number;
    matching: number;
  };
  suggestions?: Array<{
    priority: number;
    category: string;
    suggestion: string;
    example?: string;
  }>;
  optimized_resume?: string;
  match_analysis?: {
    match_score: number;
    matched_keywords: string[];
    missing_keywords: string[];
    suggestions: string[];
  };
  error?: string;
}

export interface ResumeOptimizeEvent {
  type: 'score' | 'suggestions' | 'polished' | 'complete' | 'error';
  data?: any;
  message?: string;
}


