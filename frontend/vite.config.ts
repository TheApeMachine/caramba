import babel from "@rolldown/plugin-babel";
import tailwindcss from "@tailwindcss/vite";
import { devtools } from "@tanstack/devtools-vite";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";
import { jazzPlugin } from "jazz-tools/dev/vite";
import { nitro } from "nitro/vite";
import { defineConfig } from "vite";

const config = defineConfig({
	resolve: { tsconfigPaths: true },
	json: {
		stringify: true,
	},
	plugins: [
		devtools(),
		tanstackStart(),
		react(),
		// Dev-only: starts a local Jazz server, injects VITE_JAZZ_APP_ID /
		// VITE_JAZZ_SERVER_URL, and pushes schema.ts + permissions.ts on change.
		jazzPlugin(),
		babel({
			presets: [reactCompilerPreset()],
		}),
		nitro({ rollupConfig: { external: [/^@sentry\//] } }),
		tailwindcss(),
	],
});

export default config;
