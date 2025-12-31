import * as React from "react";
import { apiUrl } from "@/lib/api";

export type RunInfo = {
	id: string;
	manifest_path: string;
	target: string;
	cmd: string[];
	cwd: string;
	pid: number;
	started_at_s: number;
	run_dir: string;
	jsonl_path: string;
	log_path: string;
	returncode: number | null;
	ended_at_s: number | null;
};

export type RunStatus =
	| "idle"
	| "starting"
	| "running"
	| "stopping"
	| "stopped"
	| "error";

export type RunLastEvent = {
	type: string | null;
	phase: string | null;
	step: number | null;
	ts: number | null;
};

export type ModelSummary = {
	type: string;
	tied_embeddings: boolean;
	vocab_size: number | null;
	d_model: number | null;
	n_heads: number | null;
	n_layers: number;
	attention_mode: string | null;
};

export type AttentionLayerSummary = {
	index: number;
	mode: string;
	d_model: number;
	n_heads: number;
	n_kv_heads: number;
	attn_dim: number | null;
	sem_dim: number | null;
	geo_dim: number | null;
	rope_enabled: boolean;
	rope_base: number;
	rope_semantic: boolean;
	tie_qk: boolean;
	null_attn: boolean;
	decoupled_gate: boolean;
};

export type LayerStat = {
	index: number;
	name: string;
	type: string;
	shape: number[] | null;
	mean_abs: number;
	rms: number;
	max_abs: number;
	mode: string;
	null_attn: boolean;
	tie_qk: boolean;
	rope_semantic: boolean;
	decoupled_gate: boolean;
};

export type VizLayer = {
	index: number;
	name: string;
	mode: string;
	n_heads?: number;
	attn?: { matrices?: number[][][]; entropy?: number[] };
	act?: { shape?: number[]; values?: number[][] };
};

type RunContextValue = {
	run: RunInfo | null;
	status: RunStatus;
	error: string | null;

	metrics: Record<string, number>;
	lastEvent: RunLastEvent;
	logLines: string[];
	logLastTs: number | null;
	serverStatus: string | null;

	selection: { manifestPath: string; target: string };
	setSelection: (s: { manifestPath: string; target: string }) => void;
	modelSummary: ModelSummary | null;
	refreshModelSummary: () => Promise<ModelSummary | null>;
	attentionLayers: AttentionLayerSummary[] | null;
	refreshAttentionLayers: () => Promise<AttentionLayerSummary[] | null>;
	layerStats: LayerStat[] | null;
	vizLayers: VizLayer[] | null;

	startRun: (args: {
		manifestPath: string;
		target: string;
	}) => Promise<RunInfo | null>;
	stopRun: () => Promise<void>;
};

const RunContext = React.createContext<RunContextValue | null>(null);

function isRecord(v: unknown): v is Record<string, unknown> {
	return typeof v === "object" && v !== null && !Array.isArray(v);
}

function asString(v: unknown): string | null {
	return typeof v === "string" ? v : null;
}

function asNumber(v: unknown): number | null {
	return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function pickNumericMetrics(v: unknown): Record<string, number> {
	if (!isRecord(v)) return {};
	const out: Record<string, number> = {};
	for (const [k, vv] of Object.entries(v)) {
		if (typeof vv === "number" && Number.isFinite(vv)) out[k] = vv;
	}
	return out;
}

export function RunProvider({ children }: { children: React.ReactNode }) {
	const [run, setRun] = React.useState<RunInfo | null>(null);
	const [status, setStatus] = React.useState<RunStatus>("idle");
	const [error, setError] = React.useState<string | null>(null);

	const [metrics, setMetrics] = React.useState<Record<string, number>>({});
	const [lastEvent, setLastEvent] = React.useState<RunLastEvent>({
		type: null,
		phase: null,
		step: null,
		ts: null,
	});
	const [logLines, setLogLines] = React.useState<string[]>([]);
	const [logLastTs, setLogLastTs] = React.useState<number | null>(null);
	const [serverStatus, setServerStatus] = React.useState<string | null>(null);

	const [selection, setSelection] = React.useState<{
		manifestPath: string;
		target: string;
	}>({
		manifestPath: "artifacts/paper/ablation_scratch.yml",
		target: "control",
	});

	const [modelSummary, setModelSummary] = React.useState<ModelSummary | null>(
		null,
	);
	const [attentionLayers, setAttentionLayers] = React.useState<
		AttentionLayerSummary[] | null
	>(null);
	const [layerStats, setLayerStats] = React.useState<LayerStat[] | null>(null);
	const [vizLayers, setVizLayers] = React.useState<VizLayer[] | null>(null);

	const refreshModelSummary =
		React.useCallback(async (): Promise<ModelSummary | null> => {
			try {
				const res = await fetch(
					apiUrl(
						`/api/manifests/model_summary?path=${encodeURIComponent(selection.manifestPath)}&target=${encodeURIComponent(selection.target)}`,
					),
				);
				const payload: unknown = await res.json();
				if (!res.ok || !isRecord(payload) || !isRecord(payload.model)) {
					setModelSummary(null);
					return null;
				}
				const m = payload.model as Record<string, unknown>;
				const nLayersRaw = m["n_layers"];
				const nLayers =
					typeof nLayersRaw === "number" && Number.isFinite(nLayersRaw)
						? Math.max(0, Math.floor(nLayersRaw))
						: 0;
				const summary: ModelSummary = {
					type: String(m["type"] ?? ""),
					tied_embeddings: Boolean(m["tied_embeddings"]),
					vocab_size:
						typeof m["vocab_size"] === "number" ? m["vocab_size"] : null,
					d_model: typeof m["d_model"] === "number" ? m["d_model"] : null,
					n_heads: typeof m["n_heads"] === "number" ? m["n_heads"] : null,
					n_layers: nLayers,
					attention_mode:
						typeof m["attention_mode"] === "string"
							? m["attention_mode"]
							: null,
				};
				setModelSummary(summary);
				return summary;
			} catch {
				setModelSummary(null);
				return null;
			}
		}, [selection.manifestPath, selection.target]);

	const refreshAttentionLayers = React.useCallback(async (): Promise<
		AttentionLayerSummary[] | null
	> => {
		try {
			const res = await fetch(
				apiUrl(
					`/api/manifests/attention_layers?path=${encodeURIComponent(selection.manifestPath)}&target=${encodeURIComponent(selection.target)}`,
				),
			);
			const payload: unknown = await res.json();
			if (!res.ok || !isRecord(payload) || !Array.isArray(payload.layers)) {
				setAttentionLayers(null);
				return null;
			}
			const out: AttentionLayerSummary[] = [];
			for (const item of payload.layers as unknown[]) {
				if (!isRecord(item)) continue;
				const idx = item.index;
				const mode = asString(item.mode) ?? "";
				if (typeof idx !== "number" || !Number.isFinite(idx)) continue;
				out.push({
					index: Math.floor(idx),
					mode,
					d_model: typeof item.d_model === "number" ? item.d_model : 0,
					n_heads: typeof item.n_heads === "number" ? item.n_heads : 0,
					n_kv_heads: typeof item.n_kv_heads === "number" ? item.n_kv_heads : 0,
					attn_dim: typeof item.attn_dim === "number" ? item.attn_dim : null,
					sem_dim: typeof item.sem_dim === "number" ? item.sem_dim : null,
					geo_dim: typeof item.geo_dim === "number" ? item.geo_dim : null,
					rope_enabled: Boolean(item.rope_enabled),
					rope_base: typeof item.rope_base === "number" ? item.rope_base : 0,
					rope_semantic: Boolean(item.rope_semantic),
					tie_qk: Boolean(item.tie_qk),
					null_attn: Boolean(item.null_attn),
					decoupled_gate: Boolean(item.decoupled_gate),
				});
			}
			setAttentionLayers(out);
			return out;
		} catch {
			setAttentionLayers(null);
			return null;
		}
	}, [selection.manifestPath, selection.target]);

	React.useEffect(() => {
		// Keep the visualization aligned with the selected manifest/target.
		void refreshModelSummary();
		void refreshAttentionLayers();
	}, [refreshModelSummary, refreshAttentionLayers]);

	React.useEffect(() => {
		if (!run?.id) return;

		const es = new EventSource(
			apiUrl(`/api/runs/${encodeURIComponent(run.id)}/events?from=end`),
		);

		es.onmessage = (evt) => {
			let parsed: unknown;
			try {
				parsed = JSON.parse(evt.data) as unknown;
			} catch {
				return;
			}
			if (!isRecord(parsed)) return;

			const t = asString(parsed["type"]);
			const phase = asString(parsed["phase"]);
			const step = asNumber(parsed["step"]);
			const ts = asNumber(parsed["ts"]);

			setLastEvent({ type: t, phase, step, ts });

			if (t === "server" && isRecord(parsed["data"])) {
				const data = parsed["data"] as Record<string, unknown>;
				const st = asString(data["status"]) ?? asString(data["error"]);
				if (st) setServerStatus(st);
			}

			if (t === "metrics" && isRecord(parsed["data"])) {
				const data = parsed["data"];
				const m = pickNumericMetrics(
					(data as Record<string, unknown>)["metrics"],
				);
				if (Object.keys(m).length > 0) {
					setMetrics(m);
					setServerStatus(null);
				}
			}

			if (t === "layer_stats" && isRecord(parsed["data"])) {
				const layers = (parsed["data"] as Record<string, unknown>)["layers"];
				if (!Array.isArray(layers)) return;
				const out: LayerStat[] = [];
				for (const item of layers) {
					if (!isRecord(item)) continue;
					const idx = item.index;
					if (typeof idx !== "number" || !Number.isFinite(idx)) continue;
					out.push({
						index: Math.floor(idx),
						name: typeof item.name === "string" ? item.name : "",
						type: typeof item.type === "string" ? item.type : "",
						shape: Array.isArray(item.shape)
							? (item.shape.filter((x) => typeof x === "number") as number[])
							: null,
						mean_abs: typeof item.mean_abs === "number" ? item.mean_abs : 0,
						rms: typeof item.rms === "number" ? item.rms : 0,
						max_abs: typeof item.max_abs === "number" ? item.max_abs : 0,
						mode: typeof item.mode === "string" ? item.mode : "",
						null_attn: Boolean(item.null_attn),
						tie_qk: Boolean(item.tie_qk),
						rope_semantic: Boolean(item.rope_semantic),
						decoupled_gate: Boolean(item.decoupled_gate),
					});
				}
				setLayerStats(out);
			}

			if (t === "viz" && isRecord(parsed["data"])) {
				const layers = (parsed["data"] as Record<string, unknown>)["layers"];
				if (!Array.isArray(layers)) return;
				const out: VizLayer[] = [];
				for (const item of layers) {
					if (!isRecord(item)) continue;
					const idx = item.index;
					if (typeof idx !== "number" || !Number.isFinite(idx)) continue;
					const attn = isRecord(item.attn) ? (item.attn as Record<string, unknown>) : null;
					const act = isRecord(item.act) ? (item.act as Record<string, unknown>) : null;
					out.push({
						index: Math.floor(idx),
						name: typeof item.name === "string" ? item.name : "",
						mode: typeof item.mode === "string" ? item.mode : "",
						n_heads: typeof item.n_heads === "number" ? item.n_heads : undefined,
						attn: attn
							? {
									matrices: Array.isArray(attn.matrices)
										? (attn.matrices as number[][][])
										: undefined,
									entropy: Array.isArray(attn.entropy)
										? (attn.entropy as number[])
										: undefined,
								}
							: undefined,
						act: act
							? {
									shape: Array.isArray(act.shape) ? (act.shape as number[]) : undefined,
									values: Array.isArray(act.values)
										? (act.values as number[][])
										: undefined,
								}
							: undefined,
					});
				}
				setVizLayers(out);
			}
		};

		es.onerror = () => {
			// Keep this soft; EventSource auto-reconnects.
		};

		return () => {
			es.close();
		};
	}, [run?.id]);

	React.useEffect(() => {
		if (!run?.id) return;

		const es = new EventSource(
			// Use from=start to avoid a race where the run prints before EventSource connects.
			// The UI still only keeps the last N lines, so this won't grow unbounded.
			apiUrl(`/api/runs/${encodeURIComponent(run.id)}/logs?from=start`),
		);

		es.onmessage = (evt) => {
			let parsed: unknown;
			try {
				parsed = JSON.parse(evt.data) as unknown;
			} catch {
				return;
			}
			if (!isRecord(parsed)) return;

			const t = asString(parsed["type"]);
			if (t !== "log") return;
			if (!isRecord(parsed["data"])) return;
			const line = asString(
				(parsed["data"] as Record<string, unknown>)["line"],
			);
			if (!line) return;

			setLogLines((prev) => {
				const next =
					prev.length >= 300 ? prev.slice(prev.length - 299) : prev.slice();
				next.push(line);
				return next;
			});
			setLogLastTs(Date.now());
		};

		es.onerror = () => {
			// Keep soft; EventSource auto-reconnects.
		};

		return () => {
			es.close();
		};
	}, [run?.id]);

	const startRun = React.useCallback(
		async ({
			manifestPath,
			target,
		}: {
			manifestPath: string;
			target: string;
		}) => {
			setError(null);
			setStatus("starting");
			setMetrics({});
			setLastEvent({ type: null, phase: null, step: null, ts: null });
			setLogLines([]);
			setLogLastTs(null);
			setServerStatus("starting");
			setLayerStats(null);
			setVizLayers(null);

			try {
				const controller = new AbortController();
				const t = window.setTimeout(() => controller.abort(), 20_000);
				const res = await fetch(apiUrl("/api/runs"), {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						manifest_path: manifestPath,
						target,
					}),
					signal: controller.signal,
				});
				window.clearTimeout(t);
				const payload: unknown = await res.json();
				if (!res.ok) {
					setStatus("error");
					setError(
						isRecord(payload) ? JSON.stringify(payload) : "Failed to start run",
					);
					return null;
				}
				if (!isRecord(payload) || !isRecord(payload.run)) {
					setStatus("error");
					setError("Malformed server response");
					return null;
				}
				const r = payload.run as RunInfo;
				setRun(r);
				setStatus("running");
				return r;
			} catch (e) {
				setStatus("error");
				if (e instanceof DOMException && e.name === "AbortError") {
					setError(
						"Start run request timed out (20s). The server may be down or blocked. Check the `caramba serve` terminal.",
					);
				} else {
					setError(e instanceof Error ? e.message : String(e));
				}
				return null;
			}
		},
		[],
	);

	const stopRun = React.useCallback(async () => {
		if (!run?.id) return;
		setError(null);
		setStatus("stopping");
		try {
			await fetch(apiUrl(`/api/runs/${encodeURIComponent(run.id)}/stop`), {
				method: "POST",
			});
			setStatus("stopped");
		} catch (e) {
			setStatus("error");
			setError(e instanceof Error ? e.message : String(e));
		}
	}, [run?.id]);

	const value: RunContextValue = React.useMemo(
		() => ({
			run,
			status,
			error,
			metrics,
			lastEvent,
			logLines,
			logLastTs,
			serverStatus,
			selection,
			setSelection,
			modelSummary,
			refreshModelSummary,
			attentionLayers,
			refreshAttentionLayers,
			layerStats,
			vizLayers,
			startRun,
			stopRun,
		}),
		[
			run,
			status,
			error,
			metrics,
			lastEvent,
			logLines,
			logLastTs,
			serverStatus,
			selection,
			modelSummary,
			refreshModelSummary,
			attentionLayers,
			refreshAttentionLayers,
			layerStats,
			vizLayers,
			startRun,
			stopRun,
		],
	);

	return <RunContext.Provider value={value}>{children}</RunContext.Provider>;
}

export function useRun(): RunContextValue {
	const ctx = React.useContext(RunContext);
	if (!ctx) {
		throw new Error("useRun must be used within <RunProvider />");
	}
	return ctx;
}
