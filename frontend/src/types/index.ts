export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

export interface StreamEvent {
  type: 'thought' | 'token' | 'complete' | 'error';
  content?: string;
  full_text?: string;
}

export interface AgentRequest {
  query: string;
  model?: string;
  provider?: 'local' | 'online';
  deepThinking?: boolean;
}

export interface UploadRequest {
  file: File;
  query: string;
  provider?: string;
}

export interface AgentState {
  query: string;
  response: string;
  thoughts: string[];
  isLoading: boolean;
  isStopped: boolean;
  error: string | null;
  uploadedFile: File | null;
  showNewChatModal: boolean;
}

export type StyleType = 'professional' | 'concise' | 'persuasive' | 'academic';

export interface AppState extends AgentState {
  showMore: boolean;
}
