import React, { createContext, useContext, useReducer } from 'react';
import type { ReactNode } from 'react';
import type { AppState } from '../types';

type Action =
  | { type: 'SET_QUERY'; payload: string }
  | { type: 'SET_RESPONSE'; payload: string }
  | { type: 'ADD_THOUGHT'; payload: string }
  | { type: 'CLEAR_THOUGHTS' }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_STOPPED'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_FILE'; payload: File | null }
  | { type: 'TOGGLE_SHOW_MORE' }
  | { type: 'TOGGLE_NEW_CHAT_MODAL' }
  | { type: 'RESET_ALL' };

const initialState: AppState = {
  query: '',
  response: '',
  thoughts: [],
  isLoading: false,
  isStopped: false,
  error: null,
  uploadedFile: null,
  showNewChatModal: false,
  showMore: false
};

function appReducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_QUERY':
      return { ...state, query: action.payload };
    case 'SET_RESPONSE':
      return { ...state, response: action.payload };
    case 'ADD_THOUGHT':
      return { ...state, thoughts: [...state.thoughts, action.payload] };
    case 'CLEAR_THOUGHTS':
      return { ...state, thoughts: [] };
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_STOPPED':
      return { ...state, isStopped: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'SET_FILE':
      return { ...state, uploadedFile: action.payload };
    case 'TOGGLE_SHOW_MORE':
      return { ...state, showMore: !state.showMore };
    case 'TOGGLE_NEW_CHAT_MODAL':
      return { ...state, showNewChatModal: !state.showNewChatModal };
    case 'RESET_ALL':
      return { ...initialState };
    default:
      return state;
  }
}

interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext(): AppContextType {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
}

export default AppContext;
