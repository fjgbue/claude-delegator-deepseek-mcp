import { readFile, stat } from 'fs/promises';
import { basename, extname } from 'path';
import { callModel } from './client.mjs';
import { listProviders, resolveApiKey, keyEnvVar, resolveModel } from './providers/registry.mjs';
import { loadConfig, TASKS } from './config.mjs';
import { buildFooter, formatCost } from './pricing.mjs';
import { color, bold, dim } from './colors.mjs';

const DELEGATE_SCHEMA = {
  type: 'object',
  properties: {
    prompt: {
      type: 'string',
      description: 'The full task/prompt to send to the delegated model. Be thorough and specific.',
    },
    task: {
      type: 'string',
      enum: TASKS,
      description:
        'What kind of work this is, so routing picks the right model: ' +
        '"read" = summarize/analyze/extract from large inputs; ' +
        '"write" = generate code or docs; ' +
        '"reason" = math, logic, architecture decisions. ' +
        'Omit to use the default model.',
    },
    model: {
      type: 'string',
      description:
        'Override the routed model. A bare model id (e.g. "deepseek-v4-flash") or ' +
        '"provider:model" for cross-provider (e.g. "openrouter:moonshotai/kimi-k2.5"). ' +
        'Prefer `task` and let routing decide.',
    },
    provider: {
      type: 'string',
      description: 'Override the active provider by id (e.g. "moonshot"). Rarely needed.',
    },
    system: {
      type: 'string',
      description: 'Optional system prompt to set context/behavior',
    },
    temperature: {
      type: 'number',
      description: 'Temperature (0-2). Lower = more deterministic. Default: 0.3',
      default: 0.3,
    },
    stream: {
      type: 'boolean',
      description:
        'Stream the response as incremental chunks instead of buffering the full output. ' +
        'Recommended for large outputs (>50K tokens) to reduce memory pressure. Default: false',
      default: false,
    },
    files: {
      type: 'array',
      items: { type: 'string' },
      description:
        'Absolute file paths to read and include in the prompt. ' +
        'The MCP server reads them directly — file contents never pass through Claude\'s context window. ' +
        'Use this instead of reading files with Read/ctx_read and embedding them in prompt.',
    },
  },
  required: ['prompt'],
};

const DELEGATE_DESCRIPTION =
  'Delegate heavy, token-intensive tasks from Claude Code to a cheaper model ' +
  '(DeepSeek, Kimi, GLM, Qwen, Grok, or any configured OpenAI-compatible provider). ' +
  'Use when: analyzing large files (>300 lines), ' +
  'multi-file codebase reviews, generating outputs >200 lines, complex reasoning, math, architecture design, ' +
  'or anytime your response would exceed ~4000 tokens. Claude orchestrates; the delegate does the heavy lifting. ' +
  'Pass `task` (read/write/reason) so routing picks the right model.';

export const TOOLS = [
  {
    name: 'delegate',
    description: DELEGATE_DESCRIPTION,
    inputSchema: DELEGATE_SCHEMA,
  },
  {
    name: 'delegate_models',
    description: 'List the configured providers and their models with context windows, output limits, and pricing',
    inputSchema: { type: 'object', properties: {} },
  },
  // v2 tool names, kept registered so existing CLAUDE.md rules and muscle
  // memory keep working. Same handlers as the canonical names above.
  {
    name: 'deepseek',
    description: 'Alias of delegate (kept for v2 compatibility). ' + DELEGATE_DESCRIPTION,
    inputSchema: DELEGATE_SCHEMA,
  },
  {
    name: 'deepseek_models',
    description: 'Alias of delegate_models (kept for v2 compatibility).',
    inputSchema: { type: 'object', properties: {} },
  },
];

const ALIASES = { deepseek: 'delegate', deepseek_models: 'delegate_models' };

// Which provider+model a delegate call runs on. Deterministic precedence:
// explicit `model` > `task` routing from delegator.json > active provider's
// default_large_model_id. Exported so tests can pin the whole truth table.
export function resolveDelegation(args, config = loadConfig()) {
  const activeId = args.provider || config.provider;
  const spec = args.model || (args.task && config.routing[args.task]) || null;
  return resolveModel(spec, activeId);
}

export async function handleToolCall(name, args) {
  switch (ALIASES[name] || name) {
    case 'delegate': {
      const config = loadConfig();
      const { provider, model } = resolveDelegation(args, config);

      // Read files server-side — bytes stay in the MCP process, never in Claude's context
      let prompt = args.prompt;
      const filePaths = Array.isArray(args.files) ? args.files.filter((p) => typeof p === 'string') : [];
      if (filePaths.length > 0) {
        // Rough guard: ~3 chars per token; leave half the context for output + prompt
        const contextWindow = (typeof model.context_window === 'number' && model.context_window > 0)
          ? model.context_window : 128_000;
        const maxFileBytes = Math.floor((contextWindow / 2) * 3);

        // The budget used to be accumulated inside the Promise.all callbacks,
        // AFTER `await stat(p)`. Two consequences, both silent: (a) until the
        // stats resolved `totalBytes` was still 0, so several files could pass
        // the check at once and the prompt could carry far more than the
        // budget; (b) WHICH files got skipped depended on stat() completion
        // order, not on the caller's files[] order, so the same call twice
        // could drop different files.
        // Fix: collect every stat first, then decide sequentially in files[]
        // order. Skipping is now deterministic and the budget actually holds.
        const sizes = await Promise.all(filePaths.map(async (p) => {
          try {
            return { p, size: (await stat(p)).size, err: null };
          } catch (e) {
            return { p, size: 0, err: e };
          }
        }));

        let totalBytes = 0;
        const sections = [];
        for (const { p, size, err } of sizes) {
          if (err) {
            sections.push(`### ${p}\n(error: ${err.message})`);
            continue;
          }
          if (totalBytes + size > maxFileBytes) {
            sections.push(`### ${p}\n(skipped: would exceed context window — ${(size / 1024).toFixed(1)}KB)`);
            continue;
          }
          totalBytes += size;
          const ext = extname(basename(p)).replace(/^\./, '');
          try {
            sections.push(`### ${p}\n\`\`\`${ext}\n${await readFile(p, 'utf8')}\n\`\`\``);
          } catch (e) {
            totalBytes -= size;
            sections.push(`### ${p}\n(error: ${e.message})`);
          }
        }
        prompt = args.prompt + '\n\n## FILES:\n\n' + sections.join('\n\n');
      }

      const result = await callModel({
        provider,
        model,
        prompt,
        system: args.system,
        temperature: args.temperature,
        stream: args.stream,
      });
      const header = [
        '',
        dim('─── claude-code-deepseek-delegator'),
        `${color('green', '◆')} ${bold('delegated to')} ${color('cyan', provider.name)} ${dim('(' + model.id + (args.task ? ' · ' + args.task : '') + ')')}`,
        '',
      ].join('\n');

      const footer = buildFooter(result, { provider, model, baseline: config.baseline });

      // Streamed responses: return each chunk as a separate content item
      if (result.streamed && Array.isArray(result.content)) {
        const items = [{ type: 'text', text: header }];
        for (const chunk of result.content) {
          items.push({ type: 'text', text: chunk });
        }
        items.push({ type: 'text', text: footer });
        return { content: items };
      }

      return {
        content: [{ type: 'text', text: header + result.content + footer }],
      };
    }
    case 'delegate_models': {
      const config = loadConfig();
      const providers = listProviders();
      const lines = [
        dim('─── claude-code-deepseek-delegator · providers ───'),
        '',
        dim(`active: ${config.provider} · routing: read→${config.routing.read} write→${config.routing.write} reason→${config.routing.reason}`),
        '',
      ];
      for (const p of providers) {
        const key = resolveApiKey(p)
          ? color('green', 'key detected')
          : dim(`needs ${keyEnvVar(p) || 'an api_key'}`);
        lines.push(`${color(p.available ? 'green' : 'yellow', '●')} ${bold(p.name)} ${dim('(' + p.id + ')')} — ${key}`);
        for (const m of p.models) {
          const ctx = (m.context_window / 1024).toFixed(0);
          const out = m.default_max_tokens ? (m.default_max_tokens / 1024).toFixed(0) + 'K out' : '';
          lines.push(`  ${bold(m.id)}  ${dim(`${ctx}K ctx  ${out}  ${formatCost(m.cost_per_1m_in ?? 0)}/${formatCost(m.cost_per_1m_out ?? 0)} per 1M`)}`);
        }
        lines.push('');
      }
      lines.push(dim('Cross-provider spec: "provider:model-id" · config: ~/.claude/delegator.json'));
      lines.push(dim('──────────────────────────────────────────────────'));
      return { content: [{ type: 'text', text: lines.join('\n') }] };
    }
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}
