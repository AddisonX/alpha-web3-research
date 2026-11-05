import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const apiTarget = env.ALPHA_DASHBOARD_API_TARGET || "http://127.0.0.1:8080";

  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 4173,
      proxy: {
        "/api": {
          target: apiTarget,
          changeOrigin: true,
        },
      },
    },
    preview: {
      host: "127.0.0.1",
      port: 4173,
      proxy: {
        "/api": {
          target: apiTarget,
          changeOrigin: true,
        },
      },
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (id.includes("node_modules/recharts") || id.includes("node_modules/d3-")) {
              return "charts";
            }
            if (id.includes("node_modules/react") || id.includes("node_modules/scheduler")) {
              return "react";
            }
            if (id.includes("node_modules/lucide-react")) {
              return "icons";
            }
            return undefined;
          }
        },
      },
    },
  };
});
