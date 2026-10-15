/**
 * Test script for CarePulse MCP Server over stdio
 */
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const serverPath = path.join(__dirname, 'dist', 'index.js');

console.log('Spawning MCP server at:', serverPath);
const proc = spawn('node', [serverPath], {
  stdio: ['pipe', 'pipe', 'inherit'],
});

let messageId = 1;

function sendRpc(method, params = {}) {
  const req = {
    jsonrpc: '2.0',
    id: messageId++,
    method,
    params,
  };
  const jsonStr = JSON.stringify(req);
  proc.stdin.write(jsonStr + '\n');
}

let buffer = '';

proc.stdout.on('data', (chunk) => {
  buffer += chunk.toString();
  const lines = buffer.split('\n');
  buffer = lines.pop(); // keep remainder

  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      const msg = JSON.parse(line);
      console.log(`[RECV ID=${msg.id || 'notif'}] Result/Method:`, msg.method || (msg.result ? 'Got Result' : 'Error'));
      if (msg.result && msg.result.tools) {
        console.log('Available MCP Tools:', msg.result.tools.map((t) => t.name));
      }
      if (msg.result && msg.result.content) {
        console.log('Tool Output:', msg.result.content[0]?.text?.slice(0, 150) + '...');
      }
    } catch (e) {
      console.log('Non-JSON line:', line);
    }
  }
});

// 1. Initialize
setTimeout(() => {
  console.log('\n--- Step 1: Initialize ---');
  sendRpc('initialize', {
    protocolVersion: '2024-11-05',
    capabilities: {},
    clientInfo: { name: 'test-client', version: '1.0.0' },
  });
}, 500);

// 2. List tools
setTimeout(() => {
  console.log('\n--- Step 2: List Tools ---');
  sendRpc('tools/list', {});
}, 1200);

// 3. Call check_red_zone (local logic + threshold check)
setTimeout(() => {
  console.log('\n--- Step 3: Call check_red_zone (occupancy_rate: 88.5) ---');
  sendRpc('tools/call', {
    name: 'check_red_zone',
    arguments: { occupancy_rate: 88.5, department: 'Cardiology' },
  });
}, 2000);

// 4. Exit
setTimeout(() => {
  console.log('\n--- All stdio RPC tests dispatched ---');
  proc.kill();
  process.exit(0);
}, 3000);
