// Provider-agnostic OpenAI-compatible chat client. Takes a provider object +
// model entry from the registry — no module-level provider state. Zero
// dependencies: node:https / node:http only.

import { request as httpsRequest } from 'node:https';
import { request as httpRequest } from 'node:http';
import { resolveApiKey, keyEnvVar } from './providers/registry.mjs';

const rawTimeout = parseInt(process.env.DELEGATOR_TIMEOUT || process.env.DEEPSEEK_TIMEOUT || '120000', 10);
const TIMEOUT_MS = rawTimeout > 0 ? rawTimeout : 120000;
const MAX_RETRIES = parseInt(process.env.DELEGATOR_MAX_RETRIES || process.env.DEEPSEEK_MAX_RETRIES || '2', 10);

class ProviderError extends Error {
  constructor(message, status, data, provider) {
    super(message);
    this.name = 'ProviderError';
    this.status = status;
    this.data = data;
    this.provider = provider;
  }
}

// v2 export name, kept so nothing importing the old class name breaks.
export { ProviderError, ProviderError as DeepSeekError };

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function callWithRetry(fn, retries = MAX_RETRIES) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (attempt < retries && (err.status >= 500 || err.status === 429)) {
        const delay = Math.min(1000 * 2 ** attempt, 16000);
        await sleep(delay);
      } else {
        throw err;
      }
    }
  }
  throw lastErr;
}

function parseUsage(u) {
  if (!u) return null;
  // Cached prompt tokens: DeepSeek reports prompt_cache_hit_tokens; the OpenAI
  // shape is prompt_tokens_details.cached_tokens. Billed at the model's cached
  // input rate by pricing.mjs.
  const cached = u.prompt_cache_hit_tokens ?? u.prompt_tokens_details?.cached_tokens ?? 0;
  return {
    promptTokens: u.prompt_tokens,
    completionTokens: u.completion_tokens,
    totalTokens: u.total_tokens,
    cachedTokens: typeof cached === 'number' ? cached : 0,
  };
}

/**
 * Parse SSE (Server-Sent Events) stream from a response.
 * Reads lines from the stream, extracts `data:` lines, parses JSON,
 * and aggregates content deltas into an array of text chunks.
 */
function parseSSEStream(res, provider) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let buffer = '';
    let model = null;
    let usage = null;
    let finishReason = 'unknown';

    // Decode across chunk boundaries. `data.toString()` decodes each Buffer
    // independently, so a multi-byte UTF-8 sequence split by TCP framing is
    // decoded as invalid bytes on BOTH sides — one 3-byte CJK character comes
    // out as 2~3 U+FFFD. The `buffer` line-join below does not undo it: by
    // then the loss is already baked into a string.
    // No flush on 'end' on purpose: the decoder holds back at most a trailing
    // incomplete sequence, and the residual `buffer` is already discarded
    // there, so a flush would be unobservable.
    const decoder = new TextDecoder('utf-8');

    res.on('data', (data) => {
      buffer += decoder.decode(data, { stream: true });
      const lines = buffer.split('\n');
      // Keep the last (potentially incomplete) line in the buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data:')) continue;

        const payload = trimmed.slice(5).trim();
        if (payload === '[DONE]') continue;

        try {
          const json = JSON.parse(payload);
          if (json.error) {
            return reject(new ProviderError(json.error.message, res.statusCode, json.error, provider.id));
          }
          const choice = json.choices?.[0];
          const delta = choice?.delta;
          if (delta?.content) {
            chunks.push(delta.content);
          }
          if (!model && json.model) model = json.model;
          if (json.usage) usage = parseUsage(json.usage);
          if (choice?.finish_reason) finishReason = choice.finish_reason;
        } catch {
          // Skip malformed SSE lines
        }
      }
    });

    let settled = false;
    const settle = (fn, val) => {
      if (!settled) { settled = true; fn(val); }
    };

    res.on('end', () => {
      settle(resolve, {
        streamed: true,
        content: chunks,
        model,
        usage,
        finishReason,
      });
    });

    res.on('error', (err) => settle(reject, err));

    // Network reset without 'end' — promise would hang forever without this
    res.on('close', () => settle(reject, new Error('Stream closed prematurely')));
  });
}

// api_endpoint (e.g. "https://api.deepseek.com/v1", "http://localhost:11434/v1")
// → node request options for POST <endpoint>/chat/completions.
function endpointOptions(provider) {
  let url;
  try {
    url = new URL(provider.api_endpoint);
  } catch {
    throw new ProviderError(`Provider ${provider.id} has an invalid api_endpoint: ${provider.api_endpoint}`, 400, null, provider.id);
  }
  return {
    requester: url.protocol === 'http:' ? httpRequest : httpsRequest,
    hostname: url.hostname,
    port: url.port || undefined,
    path: url.pathname.replace(/\/+$/, '') + '/chat/completions',
  };
}

/**
 * Call a chat model on any OpenAI-compatible provider.
 * @param {object} opts
 * @param {object} opts.provider  registry provider object
 * @param {object} opts.model     registry model entry
 * @param {string} opts.prompt
 * @param {string} [opts.system]
 * @param {number} [opts.temperature]
 * @param {number} [opts.maxTokens]
 * @param {boolean} [opts.stream]
 * @param {string} [opts.apiKey]     override (init wizard validates pasted keys before saving)
 * @param {number} [opts.timeoutMs]  override the default/env timeout
 */
export async function callModel({ provider, model, prompt, system, temperature = 0.3, maxTokens, stream = false, apiKey, timeoutMs }) {
  if (!provider || !model) throw new ProviderError('provider and model are required', 400);

  const key = apiKey || resolveApiKey(provider);
  if (!key) {
    const envVar = keyEnvVar(provider);
    throw new ProviderError(
      `API key for ${provider.name} is not set` +
      (envVar ? ` — export ${envVar}, or re-run \`npx claude-code-deepseek-delegator init\`` : ''),
      401, null, provider.id
    );
  }

  // --- Input parameter validation ---
  if (typeof prompt !== 'string' || prompt.trim().length === 0) {
    throw new ProviderError('prompt must be a non-empty string', 400, null, provider.id);
  }

  if (typeof temperature === 'number') {
    if (temperature < 0 || temperature > 2) {
      throw new ProviderError(
        `temperature must be between 0 and 2 (got ${temperature})`,
        400, null, provider.id
      );
    }
  }

  if (maxTokens !== undefined && maxTokens !== null) {
    if (!Number.isInteger(maxTokens) || maxTokens <= 0) {
      throw new ProviderError(
        `maxTokens must be a positive integer (got ${maxTokens})`,
        400, null, provider.id
      );
    }
  }
  // --- End parameter validation ---

  const messages = [];
  if (system) messages.push({ role: 'system', content: system });
  messages.push({ role: 'user', content: prompt });

  const body = {
    model: model.id,
    messages,
    temperature,
    max_tokens: maxTokens || model.default_max_tokens || 8192,
    stream: stream || undefined,
  };

  const { requester, hostname, port, path } = endpointOptions(provider);
  const timeout = timeoutMs > 0 ? timeoutMs : TIMEOUT_MS;
  const baseHeaders = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${key}`,
    ...(provider.default_headers || {}),
  };

  // Streaming path: parse SSE, return array of content chunks
  if (stream) {
    body.stream = true;
    return callWithRetry(() => {
      return new Promise((resolve, reject) => {
        const payload = JSON.stringify(body);

        const req = requester(
          {
            hostname,
            port,
            path,
            method: 'POST',
            timeout,
            headers: {
              ...baseHeaders,
              'Content-Length': Buffer.byteLength(payload),
              Accept: 'text/event-stream',
            },
          },
          (res) => {
            if (res.statusCode >= 400) {
              // Same as the non-streaming path below: `data += chunk` decodes
              // each Buffer on its own and mangles any multi-byte character
              // split across chunks. Concat first, decode once.
              const bufs = [];
              res.on('data', (chunk) => bufs.push(chunk));
              res.on('error', (err) => reject(new ProviderError(err.message, 0, null, provider.id)));
              res.on('end', () => {
                const data = Buffer.concat(bufs).toString('utf8');
                try {
                  const json = JSON.parse(data);
                  reject(new ProviderError(json.error?.message || `HTTP ${res.statusCode}`, res.statusCode, json.error, provider.id));
                } catch {
                  reject(new ProviderError(`HTTP ${res.statusCode}: ${data.slice(0, 200)}`, res.statusCode, null, provider.id));
                }
              });
              return;
            }
            parseSSEStream(res, provider).then(resolve, reject);
          }
        );

        req.on('timeout', () => {
          req.destroy();
          reject(new ProviderError(`Request timed out after ${timeout / 1000}s`, 408, null, provider.id));
        });

        req.on('error', (err) => {
          reject(new ProviderError(err.message, 0, null, provider.id));
        });

        req.write(payload);
        req.end();
      });
    });
  }

  // Non-streaming path (unchanged)
  return callWithRetry(() => {
    return new Promise((resolve, reject) => {
      const payload = JSON.stringify(body);

      const req = requester(
        {
          hostname,
          port,
          path,
          method: 'POST',
          timeout,
          headers: {
            ...baseHeaders,
            'Content-Length': Buffer.byteLength(payload),
          },
        },
        (res) => {
          // This is the default path (`stream` defaults to false in tools.mjs).
          // `data += chunk` is `string + Buffer`, which implicitly calls
          // Buffer.prototype.toString() on EACH chunk: a 3-byte CJK character
          // split by TCP framing is decoded as invalid bytes on both sides, so
          // one character becomes 2~3 U+FFFD. Concat first, decode once.
          const bufs = [];
          res.on('data', (chunk) => bufs.push(chunk));
          res.on('error', (err) => reject(new ProviderError(err.message, 0, null, provider.id)));
          res.on('end', () => {
            const data = Buffer.concat(bufs).toString('utf8');
            if (res.statusCode >= 400) {
              return reject(new ProviderError(
                `${provider.name} API error (${res.statusCode}): ${data.slice(0, 200)}`,
                res.statusCode,
                { body: data.slice(0, 1000) },
                provider.id
              ));
            }
            try {
              const json = JSON.parse(data);
              if (json.error) {
                return reject(new ProviderError(json.error.message, res.statusCode, json.error, provider.id));
              }
              const choice = json.choices?.[0];
              const message = choice?.message;
              const finish = choice?.finish_reason ?? 'unknown';
              // The model can return an empty `content` while still billing
              // tokens (e.g. the whole budget went to reasoning and finish_reason
              // came back 'length'). Surface why instead of returning a silent
              // blank that looks like the tool is broken.
              let content = typeof message?.content === 'string' ? message.content : '';
              if (!content) {
                const reasoning = typeof message?.reasoning_content === 'string' ? message.reasoning_content : '';
                if (finish === 'length') {
                  content = '[no answer returned — hit the token limit before producing output'
                    + (reasoning ? '; partial reasoning: ' + reasoning : '') + ']';
                } else {
                  content = reasoning || '(empty response)';
                }
              }
              resolve({
                content,
                model: json.model,
                usage: parseUsage(json.usage),
                finishReason: finish,
              });
            } catch (e) {
              reject(new ProviderError(`Failed to parse response: ${e.message}`, res.statusCode, data, provider.id));
            }
          });
        }
      );

      req.on('timeout', () => {
        req.destroy();
        reject(new ProviderError(`Request timed out after ${timeout / 1000}s`, 408, null, provider.id));
      });

      req.on('error', (err) => {
        reject(new ProviderError(err.message, 0, null, provider.id));
      });

      req.write(payload);
      req.end();
    });
  });
}
