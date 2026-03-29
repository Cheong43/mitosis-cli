import test from 'node:test';
import assert from 'node:assert/strict';
import { getGlobalRpmLimiter, resetGlobalRpmLimitersForTest } from '../utils/RpmLimiter.js';

test('global RPM limiter shares a single bucket across callers in the same process', async () => {
  resetGlobalRpmLimitersForTest();

  const limiterA = getGlobalRpmLimiter('llm', 6000);
  const limiterB = getGlobalRpmLimiter('llm', 6000);

  assert.equal(limiterA, limiterB);

  await limiterA.acquire();

  const startedAt = Date.now();
  await limiterB.acquire();
  const elapsedMs = Date.now() - startedAt;

  assert.ok(
    elapsedMs >= 8,
    `expected the second acquire to wait for the shared bucket, but it only waited ${elapsedMs}ms`,
  );
});

test('queued limiter tasks continue after an earlier task throws', async () => {
  resetGlobalRpmLimitersForTest();

  const limiter = getGlobalRpmLimiter('llm', 0);
  const order: string[] = [];

  await assert.rejects(
    limiter.run(async () => {
      order.push('first');
      throw new Error('boom');
    }),
    /boom/,
  );

  const second = await limiter.run(async () => {
    order.push('second');
    return 'ok';
  });

  assert.equal(second, 'ok');
  assert.deepEqual(order, ['first', 'second']);
});
