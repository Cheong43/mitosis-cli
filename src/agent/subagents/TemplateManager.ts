import * as fs from 'fs';
import * as path from 'path';
import type { SubagentTemplate } from '../../persistence/types.js';
import type { SubagentHandler } from './types.js';

export class TemplateManager {
  private templatesDir: string;
  private loadedTemplates: Map<string, SubagentTemplate> = new Map();

  constructor(private projectRoot: string) {
    this.templatesDir = path.join(projectRoot, '.mitosis', 'templates');
  }

  async loadTemplate(name: string): Promise<SubagentTemplate> {
    if (this.loadedTemplates.has(name)) {
      return this.loadedTemplates.get(name)!;
    }

    const templateDir = path.join(this.templatesDir, name);
    const configPath = path.join(templateDir, 'config.json');

    if (!fs.existsSync(configPath)) {
      throw new Error(`Template not found: ${name}`);
    }

    const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));

    const systemPromptPath = path.join(templateDir, config.systemPrompt);
    const systemPrompt = fs.existsSync(systemPromptPath)
      ? fs.readFileSync(systemPromptPath, 'utf-8')
      : '';

    const template: SubagentTemplate = {
      name: config.name,
      description: config.description,
      version: config.version,
      systemPrompt,
      skills: config.skills || [],
      tools: config.tools || [],
      outputCategory: config.outputCategory || 'research',
      contextBudget: config.contextBudget || 100000,
    };

    this.loadedTemplates.set(name, template);
    return template;
  }

  async listTemplates(): Promise<SubagentTemplate[]> {
    if (!fs.existsSync(this.templatesDir)) {
      return [];
    }

    const templates: SubagentTemplate[] = [];
    const entries = fs.readdirSync(this.templatesDir);

    for (const entry of entries) {
      const templatePath = path.join(this.templatesDir, entry);
      if (fs.statSync(templatePath).isDirectory()) {
        try {
          const template = await this.loadTemplate(entry);
          templates.push(template);
        } catch (err) {
          console.warn(`Failed to load template ${entry}:`, err);
        }
      }
    }

    return templates;
  }
}
