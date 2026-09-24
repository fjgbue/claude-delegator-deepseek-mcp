// Regression tests for PR #72: UTF-8 split across TCP chunks, and the
// files[] byte budget. The mock server cuts every body mid-character, so a
// per-chunk decode shows up as U+FFFD.

import { test, before, after } from 'node:test';
import { ok, equal, deepEqual } from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { callModel } from '../src/client.mjs';
import { readFilesWithinBudget } from '../src/tools.mjs';

const TEXT = '留出集是假设的属性'.repeat(40);

let server;
let provider;
const model = { id: 'mock', context_window: 128000, default_max_tokens: 4096 };

// Write `buf` in two reads, cut 1 byte into a 3-byte character.
function writeSplit(res, buf) {
  const first = buf.indexOf(Buffer.from('留')) + 1;
  res.write(buf.subarray(0, first));
  setTimeout(() => res.end(buf.subarray(first)), 10);
}

before(async () => {
  server = createServer((req, res) => {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      const { stream } = JSON.parse(body);
      if (stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        const sse = `data: ${JSON.stringify({ model: 'mock', choices: [{ delta: { content: TEXT } }] })}\n\n` +
          `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`;
        return writeSplit(res, Buffer.from(sse));
      }
      res.writeHead(200, { 'content-type': 'application/json' });
      writeSplit(res, Buffer.from(JSON.stringify({
        model: 'mock',
        choices: [{ message: { content: TEXT }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      })));
    });
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  provider = {
    id: 'mock', name: 'Mock', type: 'openai-compat',
    api_endpoint: `http://127.0.0.1:${server.address().port}/v1`,
  };
});

after(() => server.close());

test('non-streaming response split mid-character decodes intact', async () => {
  const r = await callModel({ provider, model, prompt: 'x', apiKey: 'sk-mock' });
  equal(r.content, TEXT);
});

test('streaming response split mid-character decodes intact', async () => {
  const r = await callModel({ provider, model, prompt: 'x', apiKey: 'sk-mock', stream: true });
  equal(r.content.join(''), TEXT);
});

test('files[] budget admits in files[] order and reports what it dropped', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'delg-budget-'));
  const paths = ['a', 'b', 'c', 'd'].map((n) => {
    const p = join(dir, `${n}.txt`);
    writeFileSync(p, n.repeat(400));
    return p;
  });
  const missing = join(dir, 'missing.txt');

  // Budget fits two 400-byte files. Run it repeatedly: the dropped set must
  // never depend on stat() timing.
  for (let i = 0; i < 20; i++) {
    const { sections, dropped } = await readFilesWithinBudget([...paths, missing], 1000);
    deepEqual(dropped.map((d) => d.path), [paths[2], paths[3], missing]);
    ok(sections[0].includes('a'.repeat(400)));
    ok(sections[1].includes('b'.repeat(400)));
    ok(sections[2].includes('skipped'));
  }
});

