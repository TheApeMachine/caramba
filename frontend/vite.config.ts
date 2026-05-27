import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import babel from "@rolldown/plugin-babel";
import tailwindcss from "@tailwindcss/vite";
import { devtools } from "@tanstack/devtools-vite";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";
import { jazzPlugin } from "jazz-tools/dev/vite";
import { nitro } from "nitro/vite";
import { defineConfig } from "vite";

// jazz-wasm is a dependency of jazz-tools, not a direct dependency here, so
// under pnpm's strict layout a deep `jazz-wasm/pkg/...` specifier won't resolve
// from this package's root. Resolve the binary's absolute path through the
// jazz-tools module context (the same approach jazzPlugin uses for the bare
// specifier) so providers/jazz.tsx can import it via `?url` and hand the served
// asset URL to Jazz as runtimeSources.wasmUrl.
const require = createRequire(import.meta.url);
const jazzWasmBinary = join(
	dirname(
		createRequire(require.resolve("jazz-tools/package.json")).resolve(
			"jazz-wasm/package.json",
		),
	),
	"pkg/jazz_wasm_bg.wasm",
);

const config = defineConfig({
	resolve: {
		tsconfigPaths: true,
		// Regex (not string) so the matched path is replaced while the `?url`
		// query is preserved, yielding an asset Vite serves in dev and emits on build.
		alias: [
			{
				find: /^jazz-wasm\/pkg\/jazz_wasm_bg\.wasm/,
				replacement: jazzWasmBinary,
			},
		],
	},
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
