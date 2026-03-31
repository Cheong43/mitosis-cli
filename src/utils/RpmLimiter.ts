/**
 * Token-bucket rate limiter that enforces a requests-per-minute (RPM) ceiling.
 *
 * Usage:
 *   const limiter = new RpmLimiter(60);   // allow 60 requests/min
 *   await limiter.acquire();              // blocks until a token is available
 *   await callLlm();
 *
 * When `rpm` is 0 or Infinity the limiter is a no-op.
 */
export class RpmLimiter {
  /** Maximum requests allowed per minute. */
  private rpm: number;
  /** Minimum milliseconds between consecutive requests. */
  private intervalMs: number;
  /** Timestamp (ms) of the last granted request. */
  private lastGranted = 0;
  /** FIFO queue of waiters (resolve callbacks). */
  private readonly queue: Array<() => void> = [];
  /** Timer handle for scheduled drain. */
  private drainTimer: ReturnType<typeof setTimeout> | null = null;
  /** FIFO dispatcher chain for queued request scheduling. */
  private dispatchTail: Promise<void> = Promise.resolve();
  /** Callback for rate limit wait notifications. */
  private onWaitCallback?: (waitMs: number, queueLength: number) => void;

  constructor(rpm: number) {
    this.rpm = 0;
    this.intervalMs = 0;
    this.configure(rpm);
  }

  setOnWaitCallback(callback: (waitMs: number, queueLength: number) => void): void {
    this.onWaitCallback = callback;
  }

  /** Returns true when the limiter is effectively disabled. */
  get disabled(): boolean {
    return this.rpm <= 0;
  }

  /** Update the configured ceiling for an existing limiter instance. */
  configure(rpm: number): void {
    this.rpm = rpm > 0 && Number.isFinite(rpm) ? rpm : 0;
    this.intervalMs = this.rpm > 0 ? 60_000 / this.rpm : 0;

    if (this.drainTimer !== null) {
      clearTimeout(this.drainTimer);
      this.drainTimer = null;
    }

    if (this.disabled) {
      while (this.queue.length > 0) {
        const waiter = this.queue.shift();
        waiter?.();
      }
      return;
    }

    if (this.queue.length > 0) {
      this.scheduleDrain();
    }
  }

  /**
   * Wait until a request slot is available.
   * Resolves immediately when the limiter is disabled.
   */
  acquire(): Promise<void> {
    if (this.disabled) return Promise.resolve();

    const now = Date.now();
    const nextAllowed = this.lastGranted + this.intervalMs;

    // Fast path: slot is available right now.
    if (now >= nextAllowed && this.queue.length === 0) {
      this.lastGranted = now;
      return Promise.resolve();
    }

    // Slow path: enqueue and schedule drain.
    const waitMs = Math.max(0, nextAllowed - now);
    if (this.onWaitCallback && this.queue.length === 0) {
      this.onWaitCallback(waitMs, this.queue.length + 1);
    }

    return new Promise<void>((resolve) => {
      this.queue.push(resolve);
      this.scheduleDrain();
    });
  }

  /**
   * Enqueue a task behind the limiter so requests are dispatched in FIFO order.
   * A task failure must not poison the queue for later tasks.
   */
  run<T>(task: () => Promise<T>): Promise<T> {
    const reservation = this.dispatchTail.then(() => this.acquire());
    this.dispatchTail = reservation.then(() => undefined, () => undefined);
    return reservation.then(task);
  }

  private scheduleDrain(): void {
    if (this.drainTimer !== null) return; // already scheduled
    const now = Date.now();
    const delay = Math.max(0, this.lastGranted + this.intervalMs - now);
    this.drainTimer = setTimeout(() => this.drain(), delay);
  }

  private drain(): void {
    this.drainTimer = null;
    const now = Date.now();
    const nextAllowed = this.lastGranted + this.intervalMs;
    if (now < nextAllowed) {
      // Woke up too early (unlikely but defensive).
      this.scheduleDrain();
      return;
    }
    const waiter = this.queue.shift();
    if (!waiter) return;
    this.lastGranted = now;
    waiter();
    if (this.queue.length > 0) {
      this.scheduleDrain();
    }
  }
}

const GLOBAL_RPM_LIMITERS_KEY = '__MITOSIS_CLI_GLOBAL_RPM_LIMITERS__';

type GlobalLimiterRegistry = typeof globalThis & {
  [GLOBAL_RPM_LIMITERS_KEY]?: Map<string, RpmLimiter>;
};

function getLimiterRegistry(): Map<string, RpmLimiter> {
  const globalRegistry = globalThis as GlobalLimiterRegistry;
  if (!globalRegistry[GLOBAL_RPM_LIMITERS_KEY]) {
    globalRegistry[GLOBAL_RPM_LIMITERS_KEY] = new Map<string, RpmLimiter>();
  }
  return globalRegistry[GLOBAL_RPM_LIMITERS_KEY]!;
}

/**
 * Returns a process-global limiter so concurrent agent instances share one RPM bucket.
 */
export function getGlobalRpmLimiter(scope: string, rpm: number): RpmLimiter {
  const registry = getLimiterRegistry();
  const key = String(scope || 'default').trim() || 'default';
  const existing = registry.get(key);
  if (existing) {
    existing.configure(rpm);
    return existing;
  }

  const limiter = new RpmLimiter(rpm);
  registry.set(key, limiter);
  return limiter;
}

export function resetGlobalRpmLimitersForTest(): void {
  getLimiterRegistry().clear();
}
