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
  private readonly rpm: number;
  /** Minimum milliseconds between consecutive requests. */
  private readonly intervalMs: number;
  /** Timestamp (ms) of the last granted request. */
  private lastGranted = 0;
  /** FIFO queue of waiters (resolve callbacks). */
  private readonly queue: Array<() => void> = [];
  /** Timer handle for scheduled drain. */
  private drainTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(rpm: number) {
    this.rpm = rpm > 0 && Number.isFinite(rpm) ? rpm : 0;
    this.intervalMs = this.rpm > 0 ? 60_000 / this.rpm : 0;
  }

  /** Returns true when the limiter is effectively disabled. */
  get disabled(): boolean {
    return this.rpm <= 0;
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
    return new Promise<void>((resolve) => {
      this.queue.push(resolve);
      this.scheduleDrain();
    });
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
