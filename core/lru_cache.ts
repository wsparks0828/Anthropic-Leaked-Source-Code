/**
 * LRU (Least Recently Used) Cache Implementation
 *
 * Bounded memory cache with automatic eviction of least-used entries.
 * Used by RubricScorer and other subsystems to prevent unbounded memory growth.
 */

/**
 * LRU Cache with bounded size and automatic eviction
 */
export class LRUCache<K, V> {
  private cache: Map<K, V> = new Map()
  private accessOrder: K[] = [] // Track access order for LRU
  private readonly maxSize: number

  constructor(maxSize: number = 10000) {
    if (maxSize < 1) {
      throw new Error('Cache size must be at least 1')
    }
    this.maxSize = maxSize
  }

  /**
   * Get value from cache (marks as recently used)
   */
  get(key: K): V | undefined {
    if (!this.cache.has(key)) {
      return undefined
    }

    // Mark as recently used by moving to end
    const index = this.accessOrder.indexOf(key)
    if (index > -1) {
      this.accessOrder.splice(index, 1)
    }
    this.accessOrder.push(key)

    return this.cache.get(key)
  }

  /**
   * Set value in cache (marks as recently used)
   */
  set(key: K, value: V): void {
    // If key already exists, remove it from access order
    if (this.cache.has(key)) {
      const index = this.accessOrder.indexOf(key)
      if (index > -1) {
        this.accessOrder.splice(index, 1)
      }
    }

    // Add new entry
    this.cache.set(key, value)
    this.accessOrder.push(key)

    // Evict least recently used if over capacity
    if (this.cache.size > this.maxSize) {
      const lruKey = this.accessOrder.shift()
      if (lruKey !== undefined) {
        this.cache.delete(lruKey)
      }
    }
  }

  /**
   * Check if key exists
   */
  has(key: K): boolean {
    return this.cache.has(key)
  }

  /**
   * Clear all entries
   */
  clear(): void {
    this.cache.clear()
    this.accessOrder = []
  }

  /**
   * Get current size
   */
  size(): number {
    return this.cache.size
  }

  /**
   * Get statistics
   */
  stats(): {
    size: number
    maxSize: number
    utilizationPercent: number
  } {
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      utilizationPercent: (this.cache.size / this.maxSize) * 100,
    }
  }
}
