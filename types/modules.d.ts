/**
 * Module type declarations for missing or conditionally loaded modules
 */

declare module 'bun:bundle' {
  export function feature(name: string): boolean
}

declare module '../services/compact/cachedMCConfig.js' {
  export function getCachedMCConfig(): {
    enabled: boolean
    keepRecent: number
    systemPromptSuggestSummaries: boolean
    supportedModels?: string[]
  }
}

declare module '../proactive/index.js' {
  export function isProactiveActive(): boolean
  export interface ProactiveModule {
    isProactiveActive(): boolean
  }
}

declare module '../tools/BriefTool/prompt.js' {
  export const BRIEF_PROACTIVE_SECTION: string | null
}

declare module '../tools/BriefTool/BriefTool.js' {
  export function isBriefEnabled(): boolean
}

declare module '../tools/DiscoverSkillsTool/prompt.js' {
  export const DISCOVER_SKILLS_TOOL_NAME: string | null
}

declare module '../services/skillSearch/featureCheck.js' {
  export function isSkillSearchEnabled(): boolean
}

declare module 'os' {
  export function type(): string
  export function version(): string
  export function release(): string
  export function arch(): string
  export function platform(): string
  export function homedir(): string
  export function hostname(): string
  export function freemem(): number
  export function totalmem(): number
  export function uptime(): number
  export function loadavg(): number[]
  export function cpus(): any[]
  export function getenv(key: string): string | undefined
  export function tmpdir(): string
}

export {}
