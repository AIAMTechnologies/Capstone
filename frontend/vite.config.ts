import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const rawTarget = (env.VITE_API_URL || "").trim();
  const hasExplicitTarget = rawTarget.length > 0;
  const normalizedTarget = hasExplicitTarget ? rawTarget.replace(/\/+$/, "") : "";
  const targetIncludesApiSuffix = normalizedTarget.toLowerCase().endsWith("/api");

  const proxyTarget = hasExplicitTarget
    ? targetIncludesApiSuffix
      ? normalizedTarget
      : `${normalizedTarget}/api`
    : "";

  const proxy = hasExplicitTarget
    ? {
        "/api": {
          target: proxyTarget,
          changeOrigin: true,
          rewrite: targetIncludesApiSuffix
            ? (path) => path.replace(/^\/api/, "")
            : undefined,
        },
      }
    : undefined;

  if (!hasExplicitTarget) {
    console.info(
      "VITE_API_URL is not defined. The dev server will proxy API requests to the same origin; set VITE_API_URL if you need to target a remote backend."
    );
  }

  return {
    plugins: [react()],
    server: {
      port: 3000,
      proxy,
    },
    build: {
      outDir: "dist",
      sourcemap: true,
    },
  };
});
