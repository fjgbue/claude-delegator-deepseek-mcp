#!/usr/bin/env node
// MCP server: delegates heavy token tasks from Claude Code to DeepSeek.
// Zero dependencies — Node.js 20+ built-ins only.
// JSON-RPC 2.0 over stdio with Content-Length framing (MCP spec compliant).

import { createRequire } from 'node:module';
import { handleToolCall, TOOLS } from './tools.mjs';
import { listProviders } from './providers/registry.mjs';
import { loadConfig } from './config.mjs';
import { parseLines } from './framing.mjs';

const PROTOCOL_VERSION = '2024-11-05';
const SERVER_NAME = 'claude-code-deepseek-delegator';
// Single source of truth: read the version from package.json so it can never
// drift from what npm published.
const SERVER_VERSION = createRequire(import.meta.url)('../package.json').version;

// CLI subcommands. With NO subcommand — which is exactly how Claude Code spawns
// this binary — we skip all of this and fall through to the MCP server below.
const SUBCOMMAND = process.argv[2];
if (['init', 'setup', 'uninstall', 'remove', 'doctor', 'help', '--help', '-h', '--version', '-v'].includes(SUBCOMMAND)) {
  process.exit(await runCli(SUBCOMMAND, process.argv.slice(3)));
}

const MAX_BUFFER_SIZE = 10 * 1024 * 1024; // 10MB — MCP messages should never exceed a few KB
let buffer = '';
let initialized = false;
let shuttingDown = false;
const inFlightRequests = new Set();

function send(msg) {
  // Newline-delimited JSON, per the MCP stdio transport spec.
  process.stdout.write(JSON.stringify(msg) + '\n');
}

function respond(id, result) {
  send({ jsonrpc: '2.0', id, result });
}

function error(id, code, message) {
  send({ jsonrpc: '2.0', id, error: { code, message: message ?? 'Internal error' } });
}

async function handle(method, id, params) {
  switch (method) {
    case 'initialize':
      if (initialized) return error(id, -32000, 'Already initialized');
      initialized = true;
      return respond(id, {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: { tools: {} },
        serverInfo: { name: SERVER_NAME, version: SERVER_VERSION },
      });

    case 'ping':
      return respond(id, {});

    case 'notifications/initialized':
      return;

    case 'tools/list':
      if (!initialized) return error(id, -32002, 'Server not initialized');
      return respond(id, { tools: TOOLS });

    case 'tools/call': {
      if (!initialized) return error(id, -32002, 'Server not initialized');
      const { name, arguments: args } = params;
      try {
        const result = await handleToolCall(name, args || {});
        return respond(id, result);
      } catch (err) {
        return error(id, -32000, err.message);
      }
    }

    default:
      if (!initialized) {
        if (id !== undefined) error(id, -32002, 'Server not initialized');
        return;
      }
      if (id !== undefined) error(id, -32601, `Method not found: ${method}`);
  }
}

// Decode stdin across chunk boundaries. Same bug as client.mjs:
// `chunk.toString()` decodes each Buffer independently, so a long non-ASCII
// prompt split by pipe framing is corrupted ON THE WAY IN, before the model
// ever sees it. Independent of the response-side corruption.
const _stdinDecoder = new TextDecoder('utf-8');

process.stdin.on('data', (chunk) => {
  buffer += _stdinDecoder.decode(chunk, { stream: true });
  if (buffer.length > MAX_BUFFER_SIZE) {
    process.stderr.write(`WARNING: buffer exceeded ${MAX_BUFFER_SIZE} bytes — flushing to prevent memory exhaustion\n`);
    buffer = '';
  }
  const { remainder } = parseLines(buffer, (msg) => {
    const promise = handle(msg.method, msg.id, msg.params || {}).catch((err) => {
      if (msg.id !== undefined) error(msg.id, -32603, err.message);
    });
    inFlightRequests.add(promise);
    promise.finally(() => inFlightRequests.delete(promise));
  });
  buffer = remainder;
});

// Graceful shutdown: wait up to 5s for in-flight requests to complete
async function gracefulShutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;
  process.stderr.write(`${signal || 'stdin end'} received, shutting down gracefully...\n`);

  if (inFlightRequests.size > 0) {
    process.stderr.write(`Waiting for ${inFlightRequests.size} in-flight request(s) to complete (max 5s)...\n`);
    const GRACEFUL_TIMEOUT = 5000;
    const timeout = new Promise((r) =>
      setTimeout(() => {
        process.stderr.write('Graceful shutdown timeout reached, exiting.\n');
        r();
      }, GRACEFUL_TIMEOUT)
    );
    await Promise.race([
      Promise.allSettled([...inFlightRequests]),
      timeout,
    ]);
  }

  process.exit(0);
}

process.stdin.on('end', () => gracefulShutdown('stdin end'));
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Startup log to stderr (not stdout — MCP uses stdout for JSON-RPC)
const startupConfig = loadConfig();
const availableProviders = listProviders().filter((p) => p.available).map((p) => p.id);
if (availableProviders.length === 0) {
  process.stderr.write(
    'WARNING: no provider API key found (e.g. DEEPSEEK_API_KEY). The server will ' +
    'start, but every delegate call will fail with a 401 until you set a key. ' +
    'Run `npx claude-code-deepseek-delegator init` to set one up.\n'
  );
}
process.stderr.write(
  `${SERVER_NAME} v${SERVER_VERSION} ready. Active provider: ${startupConfig.provider}` +
  (availableProviders.length ? ` · keys: ${availableProviders.join(', ')}` : '') + '\n'
);

// ── CLI handlers (only reached via the SUBCOMMAND branch above) ──
// Declarations are hoisted, so the early call works. Setup modules are loaded
// lazily so they never touch the hot server-startup path.
async function runCli(sub, args) {
  if (sub === 'init' || sub === 'setup') {
    const { runInit } = await import('./setup/init.mjs');
    return runInit(args);
  }
  if (sub === 'uninstall' || sub === 'remove') {
    const { runUninstall } = await import('./setup/uninstall.mjs');
    return runUninstall(args);
  }
  if (sub === 'doctor') {
    const { runDoctor } = await import('./setup/doctor.mjs');
    return runDoctor();
  }
  if (sub === '--version' || sub === '-v') {
    console.log(SERVER_VERSION);
    return 0;
  }
  printHelp();
  return 0;
}

function printHelp() {
  console.log(`
claude-code-deepseek-delegator — delegate heavy tasks from Claude Code to a cheaper model
(DeepSeek, Kimi, GLM, Qwen, Grok, Groq, OpenRouter, or any OpenAI-compatible endpoint)

Usage:
  npx claude-code-deepseek-delegator <command>

Commands:
  (no command)   Run the MCP server (this is how Claude Code launches it)
  init           Interactive setup wizard: provider, key, routing, rules, hooks
  doctor         Verify the install, resolve the config, live-fire the gate hooks
  uninstall      Cleanly remove everything init added (incl. delegator.json)
  help           Show this help
  --version      Print version

init options:
  --dry-run             Show what would change, write nothing
  --no-hooks            Skip the settings.json hooks (rules-only, softer gate)
  --yes, -y             Non-interactive, v2-identical defaults
  --provider <id>       deepseek · moonshot · zai · zhipu · alibaba-singapore · groq · xai · openrouter
  --key <key>           API key (else env var / prompt)
  --preset <name>       balanced · cheapest · max
  --mode <name>         auto (route by task) · ask (shortlist picker)
  --baseline <name>     opus · sonnet · none
`);
}
