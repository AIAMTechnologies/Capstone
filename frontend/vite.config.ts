import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

const normalizeUrl = (value: string) => value.replace(/\/+$/, '');

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const rawApiUrl = env.VITE_API_URL ? env.VITE_API_URL.trim() : '';
  const normalizedApiUrl = rawApiUrl ? normalizeUrl(rawApiUrl) : '';

  const server: Record<string, unknown> = {
    port: 3000,
  };

  if (!normalizedApiUrl || /localhost|127\.0\.0\.1/i.test(normalizedApiUrl)) {
    const proxyTarget = normalizedApiUrl
      ? normalizedApiUrl.replace(/\/api$/i, '')
      : 'http://localhost:8000';

    server.proxy = {
      '/api': {
        target: proxyTarget,
        changeOrigin: true,
        secure: false,
      },
    };
  }

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, 'src'),
      },
    },
    server,
    build: {
      outDir: 'dist',
      sourcemap: true,
    },
  };
});
