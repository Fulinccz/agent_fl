import type { ReactNode } from 'react';
import { ProfileContext, useProfileState } from './profile-context';

/** 三页共用输入档案的 Provider：挂在 App 根部，切页不丢失输入 */
export function ProfileProvider({ children }: { children: ReactNode }) {
  const value = useProfileState();
  return <ProfileContext.Provider value={value}>{children}</ProfileContext.Provider>;
}

export default ProfileProvider;
