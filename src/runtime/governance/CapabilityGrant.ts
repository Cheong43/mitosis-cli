/**
 * CapabilityGrant — validates capability declarations at subagent registration.
 *
 * The capability system is a coarse-grained access control layer that operates
 * before the fine-grained governance policy rules. A subagent must declare in
 * its manifest exactly which host capabilities it requires. The registry
 * validates these against an allowlist policy on registration.
 */

import type { CapabilityScope } from '../../types/subagent-manifest.js';

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/**
 * Host-level capability policy loaded from .mitosis/capability-policy.json.
 * If the file is absent, ALL capabilities are allowed by default (permissive).
 */
export interface CapabilityPolicy {
  /**
   * Default decision when no specific capability rule matches.
   * 'allow' (default) | 'deny'
   */
  default: 'allow' | 'deny';
  /**
   * Per-capability overrides.
   * Key = CapabilityScope, value = 'allow' | 'deny'.
   */
  capabilities?: Partial<Record<CapabilityScope, 'allow' | 'deny'>>;
  /**
   * Optional per-subagent exceptions.
   * Key = subagentId, value = array of explicitly allowed scopes.
   */
  subagentExceptions?: Record<string, CapabilityScope[]>;
}

const DEFAULT_POLICY: CapabilityPolicy = { default: 'allow' };

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

export interface CapabilityViolation {
  subagentId: string;
  scope: CapabilityScope;
  reason: string;
}

export class CapabilityGrant {
  constructor(private readonly policy: CapabilityPolicy = DEFAULT_POLICY) {}

  /**
   * Validate the required capabilities declared by a subagent manifest.
   * Returns an empty array if all are granted, or a list of violations.
   */
  validate(subagentId: string, requires: CapabilityScope[]): CapabilityViolation[] {
    if (!requires || requires.length === 0) return [];

    const exceptions = this.policy.subagentExceptions?.[subagentId] ?? [];
    const violations: CapabilityViolation[] = [];

    for (const scope of requires) {
      if (exceptions.includes(scope)) continue; // explicit exception granted

      const override = this.policy.capabilities?.[scope];
      const decision = override ?? this.policy.default;

      if (decision === 'deny') {
        violations.push({
          subagentId,
          scope,
          reason: `Capability '${scope}' is denied by the host capability policy.`,
        });
      }
    }

    return violations;
  }

  /**
   * Convenience: throws if any violations are found.
   */
  assertGranted(subagentId: string, requires: CapabilityScope[]): void {
    const violations = this.validate(subagentId, requires);
    if (violations.length > 0) {
      const msgs = violations.map((v) => `  • ${v.scope}: ${v.reason}`).join('\n');
      throw new Error(
        `CapabilityGrant: subagent '${subagentId}' requires capabilities that are not granted:\n${msgs}`,
      );
    }
  }

  /** Load from a JSON file, falling back to permissive default. */
  static fromFile(policyPath: string): CapabilityGrant {
    try {
      const fs = require('node:fs');
      const raw = fs.readFileSync(policyPath, 'utf8');
      const parsed = JSON.parse(raw) as CapabilityPolicy;
      return new CapabilityGrant(parsed);
    } catch {
      return new CapabilityGrant(DEFAULT_POLICY);
    }
  }
}
