import type { SubagentHandler, SubagentKind } from './types.js';
import { TemplateManager } from './TemplateManager.js';

export class SubagentRegistry {
  private readonly handlers = new Map<SubagentKind, SubagentHandler>();
  private templateManager?: TemplateManager;

  constructor(initialHandlers: SubagentHandler[] = [], projectRoot?: string) {
    initialHandlers.forEach((handler) => this.register(handler));
    if (projectRoot) {
      this.templateManager = new TemplateManager(projectRoot);
    }
  }

  register(handler: SubagentHandler): void {
    if (this.handlers.has(handler.kind)) {
      throw new Error(`Subagent handler already registered for kind: ${handler.kind}`);
    }
    this.handlers.set(handler.kind, handler);
  }

  get(kind: SubagentKind): SubagentHandler | undefined {
    return this.handlers.get(kind);
  }

  list(): SubagentHandler[] {
    return Array.from(this.handlers.values());
  }

  async listTemplates() {
    return this.templateManager?.listTemplates() || [];
  }

  async loadTemplate(name: string) {
    if (!this.templateManager) {
      throw new Error('TemplateManager not initialized');
    }
    return this.templateManager.loadTemplate(name);
  }
}
