import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default ({ mode }: { mode: string }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const rawTarget = env.VITE_API_URL?.trim();
  const normalizedTarget = rawTarget?.replace(/\/+$/, "");

  if (!normalizedTarget) {
    console.warn(
      "VITE_API_URL is not defined. The dev server will start without an API proxy; set VITE_API_URL (including the /api prefix) to enable proxying."
    );
  }

  let proxyTarget: string | undefined;
  let rewrite: ((path: string) => string) | undefined;

  if (normalizedTarget) {
    if (/\/api$/i.test(normalizedTarget)) {
      proxyTarget = normalizedTarget;
      rewrite = (path) => path.replace(/^\/api/, "");
    } else {
      proxyTarget = `${normalizedTarget}/api`;
    }
  }

  return defineConfig({
    plugins: [react()],
    server: {
      port: 3000,
      proxy:
        proxyTarget
          ? {
              "/api": {
                target: proxyTarget,
                changeOrigin: true,
                rewrite,
              },
            }
          : undefined,
    },
    build: {
      outDir: "dist",
      sourcemap: true,
    },
  });
};
