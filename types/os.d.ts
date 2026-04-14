/**
 * Node.js 'os' module type definitions
 */
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
