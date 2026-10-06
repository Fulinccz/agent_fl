import { useState, useCallback } from 'react';
import { skillApiClient } from '../services/skillService';
import type { SkillRouteInfo, SkillRouteState } from '../types/skill';

interface UseSkillRouterOptions {
  onRoute?: (route: SkillRouteInfo) => void;
  onError?: (error: string) => void;
}

interface UseSkillRouterReturn {
  routeState: SkillRouteState;
  routeIntent: (userInput: string) => Promise<SkillRouteInfo | null>;
  executeAuto: (userInput: string) => Promise<{ route: SkillRouteInfo; result: any } | null>;
  clearRoute: () => void;
}

const initialState: SkillRouteState = {
  currentSkill: null,
  confidence: 0,
  source: '',
  reason: '',
  isRouting: false,
  error: null
};

/**
 * Skill Router Hook
 * 
 * 提供意图识别和技能路由能力：
 * 1. routeIntent - 仅识别意图，返回路由决策信息
 * 2. executeAuto - 自动识别并执行技能
 * 3. routeState - 当前路由状态（用于UI展示）
 * 
 * 使用示例：
 * ```tsx
 * const { routeState, routeIntent, executeAuto } = useSkillRouter({
 *   onRoute: (route) => console.log('识别到技能:', route.skill)
 * });
 * 
 * // 识别意图
 * const route = await routeIntent('帮我润色简历');
 * 
 * // 自动执行
 * const { route, result } = await executeAuto('帮我润色简历');
 * ```
 */
export function useSkillRouter(options: UseSkillRouterOptions = {}): UseSkillRouterReturn {
  const [routeState, setRouteState] = useState<SkillRouteState>(initialState);

  const routeIntent = useCallback(async (userInput: string): Promise<SkillRouteInfo | null> => {
    if (!userInput.trim()) return null;

    setRouteState(prev => ({ ...prev, isRouting: true, error: null }));

    try {
      const response = await skillApiClient.routeIntent(userInput);

      if (response.error) {
        throw new Error(response.error);
      }

      const routeInfo: SkillRouteInfo = {
        skill: response.skill,
        confidence: response.confidence,
        source: response.source,
        reason: response.reason,
        params: response.params
      };

      setRouteState({
        currentSkill: routeInfo.skill,
        confidence: routeInfo.confidence,
        source: routeInfo.source,
        reason: routeInfo.reason,
        isRouting: false,
        error: null
      });

      if (options.onRoute) {
        options.onRoute(routeInfo);
      }

      return routeInfo;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '路由失败';
      setRouteState(prev => ({
        ...prev,
        isRouting: false,
        error: errorMessage
      }));

      if (options.onError) {
        options.onError(errorMessage);
      }

      return null;
    }
  }, [options]);

  const executeAuto = useCallback(async (userInput: string): Promise<{ route: SkillRouteInfo; result: any } | null> => {
    if (!userInput.trim()) return null;

    setRouteState(prev => ({ ...prev, isRouting: true, error: null }));

    try {
      const response = await skillApiClient.executeAuto(userInput);

      if (response.error) {
        throw new Error(response.error);
      }

      const routeInfo: SkillRouteInfo = response.route;

      setRouteState({
        currentSkill: routeInfo.skill,
        confidence: routeInfo.confidence,
        source: routeInfo.source,
        reason: routeInfo.reason,
        isRouting: false,
        error: null
      });

      if (options.onRoute) {
        options.onRoute(routeInfo);
      }

      return {
        route: routeInfo,
        result: response.result
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '执行失败';
      setRouteState(prev => ({
        ...prev,
        isRouting: false,
        error: errorMessage
      }));

      if (options.onError) {
        options.onError(errorMessage);
      }

      return null;
    }
  }, [options]);

  const clearRoute = useCallback(() => {
    setRouteState(initialState);
  }, []);

  return {
    routeState,
    routeIntent,
    executeAuto,
    clearRoute
  };
}

export default useSkillRouter;
