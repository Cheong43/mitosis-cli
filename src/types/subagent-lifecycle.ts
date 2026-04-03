/**
 * SubagentLifecycle — typed events emitted by the SubagentRegistry.
 *
 * Consumers can subscribe to lifecycle events to react to subagent
 * registration, activation, deactivation, and removal without coupling
 * to registry internals.
 */

import type { SubagentManifest } from './subagent-manifest.js';

// ---------------------------------------------------------------------------
// Event payloads
// ---------------------------------------------------------------------------

/** Fired when a manifest has been validated and the adapter is initialized. */
export interface SubagentRegisteredEvent {
  type: 'registered';
  manifest: SubagentManifest;
  /** ISO timestamp. */
  at: string;
}

/** Fired when a registered subagent is enabled (made visible to the planner). */
export interface SubagentEnabledEvent {
  type: 'enabled';
  subagentId: string;
  at: string;
}

/** Fired when a subagent is disabled (hidden from the planner but kept loaded). */
export interface SubagentDisabledEvent {
  type: 'disabled';
  subagentId: string;
  at: string;
}

/**
 * Fired when an updated manifest is registered over an existing one.
 * The old adapter is shut down before the new one is initialized.
 */
export interface SubagentUpgradedEvent {
  type: 'upgraded';
  subagentId: string;
  fromVersion: string;
  toVersion: string;
  at: string;
}

/**
 * Fired when a subagent is unregistered and its adapter shut down.
 * After this event the subagent id is no longer in the registry.
 */
export interface SubagentUnregisteredEvent {
  type: 'unregistered';
  subagentId: string;
  at: string;
}

/** Fired when an adapter's initialize() or any tool execution throws unexpectedly. */
export interface SubagentErrorEvent {
  type: 'error';
  subagentId: string;
  phase: 'initialize' | 'execute' | 'shutdown' | 'mcp-connect';
  error: string;
  at: string;
}

// ---------------------------------------------------------------------------
// Union and listener type
// ---------------------------------------------------------------------------

export type SubagentLifecycleEvent =
  | SubagentRegisteredEvent
  | SubagentEnabledEvent
  | SubagentDisabledEvent
  | SubagentUpgradedEvent
  | SubagentUnregisteredEvent
  | SubagentErrorEvent;

export type SubagentLifecycleListener = (event: SubagentLifecycleEvent) => void;
