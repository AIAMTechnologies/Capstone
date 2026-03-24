import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
process.chdir(resolve(__dirname, 'frontend'));

// Pass PORT env var as --port arg so vite binds to the assigned port
const port = process.env.PORT;
if (port) {
  process.argv.push('--port', port);
}

await import('./frontend/node_modules/vite/dist/node/cli.js');
