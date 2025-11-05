import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const rawTarget = env.VITE_API_URL?.trim();
  const normalizedTarget = rawTarget ? rawTarget.replace(/\/+$/, "") : undefined;

  const usingExplicitTarget = Boolean(normalizedTarget);
  const targetIncludesApiSuffix = normalizedTarget ? /\/api$/i.test(normalizedTarget) : false;

  const proxyTarget = normalizedTarget
    ? targetIncludesApiSuffix
      ? normalizedTarget
      : `${normalizedTarget}/api`
    : undefined;

  const proxyConfig = proxyTarget
    ? {
        "/api": {
          target: proxyTarget,
          changeOrigin: true,
          rewrite: targetIncludesApiSuffix
            ? (path: string) => path.replace(/^\/api/, "")
            : undefined,
        },
      }
    : undefined;

  if (!usingExplicitTarget) {
    console.info(
      "VITE_API_URL is not defined. The dev server will proxy API requests to the same origin; set VITE_API_URL if you need to target a remote backend."
    );
  }

  return {
    plugins: [react()],
    server: {
      port: 3000,
      proxy: proxyConfig,
export default ({ mode }: { mode: string }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_API_URL;

  if (!target) {
    throw new Error(
      "VITE_API_URL is not defined. Set it to the backend API base URL (including the /api prefix)."
    );
  }

  const normalizedTarget = target.replace(/\/+$/, "");

  return defineConfig({
    plugins: [react()],
    server: {
      port: 3000,
      proxy: {
        "/api": {
          target: normalizedTarget,
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: "dist",
      sourcemap: true,
    },
  };
});
  });
};
