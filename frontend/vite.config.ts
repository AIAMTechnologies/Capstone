import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

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
  });
};
