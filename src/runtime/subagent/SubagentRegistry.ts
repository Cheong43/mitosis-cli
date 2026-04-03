/**
 * SubagentRegistry — the central hub of the subagent ecosystem.
 *
 * Responsibilities:
 *  • registerFromManifest(): validate + initialize + sandbox a subagent
 *  • discover(): scan .mitosis/agents/ directories for manifests
 *  • resolve(): look up subagents by id (+ optional semver)
 *  • listCapabilities(): aggregate tools/skills across all sandboxes
 *  • getDependencyOrder(): topological sort for safe initialization
 *  • emit lifecycle events
 *
 * The registry does NOT call AgentRuntime or break any existing subagent
 * handler contracts — the legacy SubagentHandlers continue to work through
 * the LegacyHandlerAdapter shim.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

import type { SubagentManifest } from '../../types/subagent-manifest.js';
import type { SubagentAdapter, SubagentConfig } from '../../types/subagent-adapter.js';
import type { SubagentLifecycleEvent, SubagentLifecycleListener } from '../../types/subagent-lifecycle.js';
import type { ToolDefinition } from '../tools/types.js';
import type { SkillRecord } from '../../skills/router.js';
import { createTransport } from '../transport/index.js';
import { SubagentSandbox } from './SubagentSandbox.js';
import { CapabilityGrant } from '../governance/CapabilityGrant.js';
import { GovernanceRuntime } from '../governance/GovernanceRuntime.js';

// ---------------------------------------------------------------------------
// Semver comparison (minimal — no pre-release support needed)
// ---------------------------------------------------------------------------

function parseSemver(v: string): [number, number, number] {
  const parts = v.split('.').map(Number);
  return [parts[0] ?? 0, parts[1] ?? 0, parts[2] ?? 0];
}

function semverGte(a: string, b: string): boolean {
  const [am, an, ap] = parseSemver(a);
  const [bm, bn, bp] = parseSemver(b);
  if (am !== bm) return am > bm;
  if (an !== bn) return an > bn;
  return ap >= bp;
}

// ---------------------------------------------------------------------------
// Registry entry
// ---------------------------------------------------------------------------

interface RegistryEntry {
  manifest: SubagentManifest;
  sandbox: SubagentSandbox;
}

// ---------------------------------------------------------------------------
// SubagentRegistry
// ---------------------------------------------------------------------------

export interface SubagentRegistryOptions {
  projectRoot: string;
  globalGovernance: GovernanceRuntime;
  capabilityGrant?: CapabilityGrant;
  /** If true, tools from different subagents are namespaced as "<id>/<tool>". Default true. */
  namespaceTools?: boolean;
}

export class SubagentRegistry {
  private readonly entries = new Map<string, RegistryEntry>();
  private readonly listeners: SubagentLifecycleListener[] = [];
  private readonly opts: SubagentRegistryOptions;
  private readonly capabilityGrant: CapabilityGrant;

  constructor(opts: SubagentRegistryOptions) {
    this.opts = opts;
    this.capabilityGrant = opts.capabilityGrant ?? new CapabilityGrant();
  }

  // ---------------------------------------------------------------------------
  // Lifecycle event bus
  // ---------------------------------------------------------------------------

  on(listener: SubagentLifecycleListener): void {
    this.listeners.push(listener);
  }

  off(listener: SubagentLifecycleListener): void {
    const idx = this.listeners.indexOf(listener);
    if (idx !== -1) this.listeners.splice(idx, 1);
  }

  private emit(event: SubagentLifecycleEvent): void {
    for (const l of this.listeners) {
      try { l(event); } catch { /* listeners must not throw */ }
    }
  }

  // ---------------------------------------------------------------------------
  // Registration
  // ---------------------------------------------------------------------------

  /**
   * Register a subagent from its manifest.
   *
   * Steps:
   *  1. Validate capability requirements.
   *  2. If an older version is already registered, unregister it first (upgrade).
   *  3. Create transport → connect → create adapter.
   *  4. Wrap in a sandbox and initialize.
   *  5. Emit 'registered' event.
   */
  async registerFromManifest(
    manifest: SubagentManifest,
    config: Omit<SubagentConfig, 'manifest'>,
  ): Promise<void> {
    const { id, version } = manifest;

    // 1. Validate capabilities.
    this.capabilityGrant.assertGranted(id, manifest.requires ?? []);

    // 2. Handle upgrade.
    const existing = this.entries.get(id);
    if (existing) {
      const oldVersion = existing.manifest.version;
      if (semverGte(oldVersion, version)) {
        throw new Error(
          `SubagentRegistry: subagent '${id}' version ${oldVersion} is already registered (>= ${version}). ` +
          'Unregister it first or provide a higher version.',
        );
      }
      await this.unregister(id);
      this.emit({ type: 'upgraded', subagentId: id, fromVersion: oldVersion, toVersion: version, at: new Date().toISOString() });
    }

    // 3. Create transport + adapter.
    const transport = createTransport(manifest);
    let adapter: SubagentAdapter;
    try {
      adapter = await transport.connect();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      this.emit({ type: 'error', subagentId: id, phase: 'initialize', error: msg, at: new Date().toISOString() });
      throw e;
    }

    // 4. Build sandbox + initialize.
    const sandbox = new SubagentSandbox({
      manifest,
      adapter,
      globalGovernance: this.opts.globalGovernance,
      namespaceTools: this.opts.namespaceTools,
    });

    const fullConfig: SubagentConfig = { ...config, manifest };
    try {
      await sandbox.initialize(fullConfig);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      this.emit({ type: 'error', subagentId: id, phase: 'initialize', error: msg, at: new Date().toISOString() });
      await transport.disconnect().catch(() => {});
      throw e;
    }

    // 5. Store and emit.
    this.entries.set(id, { manifest, sandbox });
    this.emit({ type: 'registered', manifest, at: new Date().toISOString() });
  }

  /**
   * Unregister a subagent and shut down its sandbox.
   */
  async unregister(id: string): Promise<void> {
    const entry = this.entries.get(id);
    if (!entry) return;
    await entry.sandbox.shutdown().catch(() => {});
    this.entries.delete(id);
    this.emit({ type: 'unregistered', subagentId: id, at: new Date().toISOString() });
  }

  // ---------------------------------------------------------------------------
  // Enable / Disable
  // ---------------------------------------------------------------------------

  enable(id: string): void {
    const entry = this.entries.get(id);
    if (!entry) throw new Error(`SubagentRegistry: subagent '${id}' not found`);
    entry.manifest.enabled = true;
    this.emit({ type: 'enabled', subagentId: id, at: new Date().toISOString() });
  }

  disable(id: string): void {
    const entry = this.entries.get(id);
    if (!entry) throw new Error(`SubagentRegistry: subagent '${id}' not found`);
    entry.manifest.enabled = false;
    this.emit({ type: 'disabled', subagentId: id, at: new Date().toISOString() });
  }

  // ---------------------------------------------------------------------------
  // Lookup
  // ---------------------------------------------------------------------------

  resolve(id: string, minVersion?: string): SubagentSandbox | undefined {
    const entry = this.entries.get(id);
    if (!entry) return undefined;
    if (minVersion && !semverGte(entry.manifest.version, minVersion)) return undefined;
    return entry.sandbox;
  }

  getManifest(id: string): SubagentManifest | undefined {
    return this.entries.get(id)?.manifest;
  }

  list(enabledOnly = false): SubagentManifest[] {
    const all = Array.from(this.entries.values()).map((e) => e.manifest);
    return enabledOnly ? all.filter((m) => m.enabled !== false) : all;
  }

  // ---------------------------------------------------------------------------
  // Capability aggregation
  // ---------------------------------------------------------------------------

  /**
   * Returns all tools exposed by enabled subagents, merged into a flat list.
   * Tools are namespaced if the registry was configured with namespaceTools=true.
   */
  listAllTools(): ToolDefinition[] {
    const tools: ToolDefinition[] = [];
    for (const { manifest, sandbox } of this.entries.values()) {
      if (manifest.enabled === false) continue;
      tools.push(...sandbox.getExposedTools());
    }
    return tools;
  }

  /**
   * Returns all skills exposed by enabled subagents.
   */
  listAllSkills(): SkillRecord[] {
    const skills: SkillRecord[] = [];
    for (const { manifest, sandbox } of this.entries.values()) {
      if (manifest.enabled === false) continue;
      skills.push(...sandbox.getExposedSkills());
    }
    return skills;
  }

  // ---------------------------------------------------------------------------
  // Dependency graph
  // ---------------------------------------------------------------------------

  /**
   * Returns registered subagent ids in topological order (dependencies first).
   * Throws on cycles.
   */
  getDependencyOrder(): string[] {
    const ids = Array.from(this.entries.keys());
    const visited = new Set<string>();
    const sorted: string[] = [];

    const visit = (id: string, stack: Set<string>): void => {
      if (visited.has(id)) return;
      if (stack.has(id)) throw new Error(`SubagentRegistry: dependency cycle detected involving '${id}'`);
      stack.add(id);
      const manifest = this.entries.get(id)?.manifest;
      for (const dep of manifest?.dependencies ?? []) {
        visit(dep, stack);
      }
      stack.delete(id);
      visited.add(id);
      sorted.push(id);
    };

    for (const id of ids) visit(id, new Set());
    return sorted;
  }

  // ---------------------------------------------------------------------------
  // Auto-discovery
  // ---------------------------------------------------------------------------

  /**
   * Scan .mitosis/agents/ directories for manifest.json files and register them.
   * Directories: <projectRoot>/.mitosis/agents/ and ~/.mitosis/agents/
   */
  async discover(config: Omit<SubagentConfig, 'manifest'>): Promise<string[]> {
    const roots = this.resolveAgentRoots();
    const registered: string[] = [];

    for (const root of roots) {
      if (!fs.existsSync(root)) continue;
      const dirs = fs.readdirSync(root, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => path.join(root, d.name));

      for (const dir of dirs) {
        const manifestPath = path.join(dir, 'manifest.json');
        if (!fs.existsSync(manifestPath)) continue;
        let manifest: SubagentManifest;
        try {
          manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as SubagentManifest;
        } catch (e) {
          console.warn(`SubagentRegistry.discover: failed to parse ${manifestPath}: ${e}`);
          continue;
        }
        try {
          await this.registerFromManifest(manifest, config);
          registered.push(manifest.id);
        } catch (e) {
          console.warn(`SubagentRegistry.discover: failed to register '${manifest.id}': ${e}`);
        }
      }
    }

    return registered;
  }

  private resolveAgentRoots(): string[] {
    const roots: string[] = [
      path.join(this.opts.projectRoot, '.mitosis', 'agents'),
    ];
    const home = process.env.HOME ?? process.env.USERPROFILE;
    if (home) roots.push(path.join(home, '.mitosis', 'agents'));
    return roots;
  }

  // ---------------------------------------------------------------------------
  // Graceful shutdown
  // ---------------------------------------------------------------------------

  async shutdownAll(): Promise<void> {
    const ids = Array.from(this.entries.keys());
    await Promise.all(ids.map((id) => this.unregister(id)));
  }
}
