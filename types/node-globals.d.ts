/**
 * Global declarations for Node.js APIs and build-time injected globals
 * This file provides type definitions when @types/node is not available
 */

// Extend global scope with necessary declarations
declare global {
  /**
   * Process global object with minimal required properties
   */
  var process: {
    env: Record<string, string | undefined>
    platform?: string
    arch?: string
    cwd?(): string
  }

  /**
   * Build-time injected MACRO object containing version and feedback info
   */
  var MACRO: {
    VERSION: string
    BUILD_TIME?: string
    ISSUES_EXPLAINER?: string
    FEEDBACK_CHANNEL?: string
    PACKAGE_URL?: string
  }

  /**
   * Global require function for dynamic CommonJS imports
   */
  function require(id: string): unknown
}

/**
 * Module type definitions
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
  export const proactive: {
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

// Ensure this file is treated as a module
export {}


