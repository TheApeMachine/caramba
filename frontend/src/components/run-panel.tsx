import * as React from "react";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardFooter,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useRun } from "@/lib/run-context";
import { apiUrl } from "@/lib/api";

export function RunPanel() {
	const {
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
		attentionLayers,
		startRun,
		stopRun,
	} = useRun();

	const manifestPath = selection.manifestPath;
	const target = selection.target;
	const [manifestTargets, setManifestTargets] = React.useState<string[] | null>(
		null,
	);
	const [manifestTargetsPath, setManifestTargetsPath] = React.useState<
		string | null
	>(null);
	const [manifestLoadError, setManifestLoadError] = React.useState<
		string | null
	>(null);
	const [targetsLoading, setTargetsLoading] = React.useState(false);
	const [targetBadges, setTargetBadges] = React.useState<
		Record<
			string,
			{
				mode: string;
				null_attn: boolean;
				tie_qk: boolean;
				rope_semantic: boolean;
				decoupled_gate: boolean;
			}
		>
	>({});
	const [showLogs, setShowLogs] = React.useState(true);
	const suiteAbortRef = React.useRef(false);
	const [suiteRunning, setSuiteRunning] = React.useState(false);
	const [suiteIdx, setSuiteIdx] = React.useState<number>(0);
	const [suiteCurrentTarget, setSuiteCurrentTarget] = React.useState<
		string | null
	>(null);
	const [suiteCurrentRunId, setSuiteCurrentRunId] = React.useState<
		string | null
	>(null);
	const [suiteHistory, setSuiteHistory] = React.useState<
		Array<{
			target: string;
			runId: string;
			returncode: number | null;
			logPath: string;
			jsonlPath: string;
		}>
	>([]);

	const lastManifestPathRef = React.useRef<string>(manifestPath);

	// Important: if the manifest path changes, any previously loaded targets/badges
	// are stale. Reset so "Run all" can’t accidentally use the wrong target list.
	React.useEffect(() => {
		if (lastManifestPathRef.current === manifestPath) return;
		lastManifestPathRef.current = manifestPath;

		setManifestTargets(null);
		setManifestTargetsPath(null);
		setTargetBadges({});
		setSuiteHistory([]);
		setSuiteIdx(0);
		setSuiteCurrentTarget(null);
		setSuiteCurrentRunId(null);
		setManifestLoadError(null);
		setTargetsLoading(false);
		setSuiteRunning(false);
		suiteAbortRef.current = false;
	}, [manifestPath]);

	const canStart =
		status === "idle" || status === "stopped" || status === "error";
	const canStop = status === "running" || status === "starting";

	const topMetrics = React.useMemo(() => {
		const preferred = ["loss", "ppl", "tok_s", "ms_step", "lr", "grad_norm"];
		const rows: Array<[string, number]> = [];
		for (const k of preferred) {
			const v = metrics[k];
			if (typeof v === "number") rows.push([k, v]);
		}
		return rows;
	}, [metrics]);

	const attentionSummary = React.useMemo(() => {
		const flags = {
			mode: "—",
			sem_dim: null as number | null,
			geo_dim: null as number | null,
			attn_dim: null as number | null,
			null_attn: false,
			tie_qk: false,
			rope_semantic: false,
			decoupled_gate: false,
		};
		if (!attentionLayers || attentionLayers.length === 0) return flags;
		const first = attentionLayers[0];
		if (first) {
			flags.mode = first.mode || "—";
			flags.sem_dim = first.sem_dim ?? null;
			flags.geo_dim = first.geo_dim ?? null;
			flags.attn_dim = first.attn_dim ?? null;
		}
		for (const a of attentionLayers) {
			flags.null_attn = flags.null_attn || Boolean(a.null_attn);
			flags.tie_qk = flags.tie_qk || Boolean(a.tie_qk);
			flags.rope_semantic = flags.rope_semantic || Boolean(a.rope_semantic);
			flags.decoupled_gate = flags.decoupled_gate || Boolean(a.decoupled_gate);
		}
		return flags;
	}, [attentionLayers]);

	const computeFlags = React.useCallback(
		(
			layers: Array<{
				mode: string;
				null_attn: boolean;
				tie_qk: boolean;
				rope_semantic: boolean;
				decoupled_gate: boolean;
			}>,
		) => {
			const flags = {
				mode: "—",
				null_attn: false,
				tie_qk: false,
				rope_semantic: false,
				decoupled_gate: false,
			};
			if (layers.length === 0) return flags;
			flags.mode = layers[0]?.mode || "—";
			for (const a of layers) {
				flags.null_attn = flags.null_attn || Boolean(a.null_attn);
				flags.tie_qk = flags.tie_qk || Boolean(a.tie_qk);
				flags.rope_semantic = flags.rope_semantic || Boolean(a.rope_semantic);
				flags.decoupled_gate =
					flags.decoupled_gate || Boolean(a.decoupled_gate);
			}
			return flags;
		},
		[],
	);

	const loadTargets = React.useCallback(async (): Promise<string[] | null> => {
		setManifestLoadError(null);
		setTargetsLoading(true);
		try {
			const res = await fetch(
				apiUrl(`/api/manifests/targets?path=${encodeURIComponent(manifestPath)}`),
			);
			const payload: unknown = await res.json();
			if (!res.ok || typeof payload !== "object" || payload === null) {
				setManifestTargets(null);
				setManifestLoadError(
					"Failed to load targets (server/proxy not reachable).",
				);
				return null;
			}
			const p = payload as { targets?: unknown; model_targets?: unknown };
			const tPreferred = p.model_targets ?? p.targets;
			if (Array.isArray(tPreferred) && tPreferred.every((x) => typeof x === "string")) {
				const out = tPreferred as string[];
				setManifestTargets(out);
				setManifestTargetsPath(manifestPath);

				// Prefetch per-target DBA flags for badges.
				try {
					const results = await Promise.all(
						out.map(async (name) => {
							const u = `/api/manifests/attention_layers?path=${encodeURIComponent(manifestPath)}&target=${encodeURIComponent(name)}`;
							const r = await fetch(apiUrl(u));
							const p: unknown = await r.json();
							if (!r.ok || typeof p !== "object" || p === null) {
								return { name, flags: null as null };
							}
							const layers = (p as { layers?: unknown }).layers;
							if (!Array.isArray(layers)) {
								return { name, flags: null as null };
							}
							const parsed = layers
								.filter(
									(x): x is Record<string, unknown> =>
										typeof x === "object" && x !== null,
								)
								.map((x) => ({
									mode: typeof x.mode === "string" ? x.mode : "—",
									null_attn: Boolean(x.null_attn),
									tie_qk: Boolean(x.tie_qk),
									rope_semantic: Boolean(x.rope_semantic),
									decoupled_gate: Boolean(x.decoupled_gate),
								}));
							return { name, flags: computeFlags(parsed) };
						}),
					);
					setTargetBadges((prev) => {
						const next = { ...prev };
						for (const r of results) {
							if (r.flags) next[r.name] = r.flags;
						}
						return next;
					});
				} catch {
					// ignore; badges are best-effort
				}

				setTargetsLoading(false);
				return out;
			}
			setManifestTargets(null);
			setManifestTargetsPath(null);
			setManifestLoadError("Manifest response did not include a targets list.");
			setTargetsLoading(false);
			return null;
		} catch {
			setManifestTargets(null);
			setManifestTargetsPath(null);
			setManifestLoadError("Failed to load targets (network error).");
			setTargetsLoading(false);
			return null;
		}
	}, [manifestPath, computeFlags]);

	const pollRunUntilDone = React.useCallback(async (runId: string) => {
		// Poll /api/runs/<id> until returncode is non-null.
		for (;;) {
			if (suiteAbortRef.current) return;
			try {
				const res = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
				const payload: unknown = await res.json();
				if (
					res.ok &&
					typeof payload === "object" &&
					payload &&
					"run" in payload
				) {
					const r = (payload as { run: unknown }).run;
					if (r && typeof r === "object") {
						const rc = (r as { returncode?: unknown }).returncode;
						if (typeof rc === "number") return;
					}
				}
			} catch {
				// ignore; keep polling
			}
			await new Promise((resolve) => setTimeout(resolve, 2000));
		}
	}, []);

	return (
		<Card size="sm" className="w-full">
			<CardHeader className="pb-3">
				<CardTitle>Caramba Run</CardTitle>
			</CardHeader>
			<CardContent className="space-y-3">
				<div className="space-y-2">
					<div className="text-xs text-muted-foreground">Manifest</div>
					<Input
						value={manifestPath}
						onChange={(e) =>
							setSelection({
								manifestPath: e.target.value,
								target: selection.target,
							})
						}
						placeholder="config/presets/ui_demo.yml"
					/>
					<div className="flex gap-2">
						<Button
							variant="outline"
							size="sm"
							type="button"
							onClick={loadTargets}
						>
							{targetsLoading ? "Loading…" : "Load targets"}
						</Button>
						<Button
							variant="outline"
							size="sm"
							type="button"
							disabled={suiteRunning}
							onClick={async () => {
								suiteAbortRef.current = false;
								setSuiteRunning(true);
								setSuiteIdx(0);
								setSuiteHistory([]);
								setSuiteCurrentTarget(null);
								setSuiteCurrentRunId(null);

								const targets =
									manifestTargets && manifestTargetsPath === manifestPath
										? manifestTargets
										: await loadTargets();
								if (!targets || targets.length === 0) {
									setSuiteRunning(false);
									return;
								}

								for (let i = 0; i < targets.length; i++) {
									if (suiteAbortRef.current) break;
									const t = targets[i];
									setSuiteIdx(i);
									setSelection({ manifestPath, target: t });
									setSuiteCurrentTarget(t);
									setSuiteCurrentRunId(null);

									const started = await startRun({ manifestPath, target: t });
									if (!started?.id) {
										// startRun should have set an error; stop the suite.
										break;
									}
									setSuiteCurrentRunId(started.id);

									await pollRunUntilDone(started.id);
									if (suiteAbortRef.current) break;

									// Fetch final status snapshot for returncode and paths.
									try {
										const res = await fetch(
											`/api/runs/${encodeURIComponent(started.id)}`,
										);
										const payload: unknown = await res.json();
										let rc: number | null = null;
										let logPath = started.log_path;
										let jsonlPath = started.jsonl_path;
										if (res.ok && payload && typeof payload === "object") {
											const rr = (payload as { run?: unknown }).run;
											if (rr && typeof rr === "object") {
												const v = (rr as { returncode?: unknown }).returncode;
												if (typeof v === "number") rc = v;
												const lp = (rr as { log_path?: unknown }).log_path;
												const jp = (rr as { jsonl_path?: unknown }).jsonl_path;
												if (typeof lp === "string") logPath = lp;
												if (typeof jp === "string") jsonlPath = jp;
											}
										}
										setSuiteHistory((prev) => [
											...prev,
											{
												target: t,
												runId: started.id,
												returncode: rc,
												logPath,
												jsonlPath,
											},
										]);
									} catch {
										setSuiteHistory((prev) => [
											...prev,
											{
												target: t,
												runId: started.id,
												returncode: null,
												logPath: started.log_path,
												jsonlPath: started.jsonl_path,
											},
										]);
									}
								}

								setSuiteRunning(false);
							}}
						>
							Run all (sequential)
						</Button>
						{suiteRunning && (
							<Button
								variant="destructive"
								size="sm"
								type="button"
								onClick={() => {
									suiteAbortRef.current = true;
								}}
							>
								Cancel suite
							</Button>
						)}
					</div>
					{manifestTargets && manifestTargets.length > 0 && (
						<div className="flex flex-wrap gap-1">
							{manifestTargets.map((t) => (
								<Button
									key={t}
									variant="outline"
									size="sm"
									type="button"
									onClick={() => setSelection({ manifestPath, target: t })}
								>
									<span className="flex items-center gap-2">
										<span>{t}</span>
										{targetBadges[t] && (
											<span className="flex items-center gap-1 text-[10px] text-slate-300">
												<span className="text-slate-500">
													{targetBadges[t].mode}
												</span>
												{targetBadges[t].null_attn && (
													<span className="px-1 rounded bg-[#e0af68]/20 text-[#e0af68] border border-[#e0af68]/30">
														null
													</span>
												)}
												{targetBadges[t].tie_qk && (
													<span className="px-1 rounded bg-[#60a5fa]/20 text-[#60a5fa] border border-[#60a5fa]/30">
														tie_qk
													</span>
												)}
												{targetBadges[t].rope_semantic && (
													<span className="px-1 rounded bg-[#bb9af7]/20 text-[#bb9af7] border border-[#bb9af7]/30">
														rope_sem
													</span>
												)}
												{targetBadges[t].decoupled_gate && (
													<span className="px-1 rounded bg-[#f472b6]/20 text-[#f472b6] border border-[#f472b6]/30">
														gate
													</span>
												)}
											</span>
										)}
									</span>
								</Button>
							))}
						</div>
					)}
					{manifestLoadError && (
						<div className="rounded-md border border-destructive/40 bg-destructive/10 px-2 py-1 text-xs text-destructive">
							{manifestLoadError}
						</div>
					)}
					{modelSummary && (
						<div className="text-xs text-muted-foreground">
							model: layers={modelSummary.n_layers} d_model=
							{modelSummary.d_model ?? "—"} heads=
							{modelSummary.n_heads ?? "—"} mode=
							{modelSummary.attention_mode ?? "—"}
						</div>
					)}
					{attentionLayers && attentionLayers.length > 0 && (
						<div className="text-xs text-muted-foreground">
							attn: {attentionSummary.mode}
							{attentionSummary.sem_dim !== null ||
							attentionSummary.geo_dim !== null ? (
								<>
									{" "}
									(sem/geo {attentionSummary.sem_dim ?? "—"}/
									{attentionSummary.geo_dim ?? "—"})
								</>
							) : null}
							{attentionSummary.attn_dim !== null ? (
								<> • attn_dim {attentionSummary.attn_dim}</>
							) : null}
							{attentionSummary.null_attn ? " • null" : ""}
							{attentionSummary.tie_qk ? " • tie_qk" : ""}
							{attentionSummary.rope_semantic ? " • rope_sem" : ""}
							{attentionSummary.decoupled_gate ? " • gate" : ""}
						</div>
					)}
					{suiteRunning && manifestTargets && manifestTargets.length > 0 && (
						<div className="text-xs text-muted-foreground">
							Suite running: {suiteIdx + 1}/{manifestTargets.length}
							{suiteCurrentTarget ? ` • ${suiteCurrentTarget}` : ""}
							{suiteCurrentRunId
								? ` • run ${suiteCurrentRunId.slice(0, 8)}`
								: ""}
						</div>
					)}
					{suiteHistory.length > 0 && (
						<div className="rounded-md border border-border p-2">
							<div className="text-xs text-muted-foreground mb-2">
								Suite history
							</div>
							<div className="space-y-1 text-xs">
								{suiteHistory.map((h) => (
									<div key={h.runId} className="flex flex-col gap-0.5">
										<div className="flex items-center justify-between">
											<div className="font-medium">{h.target}</div>
											<div className="font-mono">
												rc={h.returncode === null ? "?" : h.returncode}
											</div>
										</div>
										<div className="text-muted-foreground font-mono break-all">
											log: {h.logPath}
										</div>
										<div className="text-muted-foreground font-mono break-all">
											jsonl: {h.jsonlPath}
										</div>
									</div>
								))}
							</div>
						</div>
					)}
				</div>
				<div className="space-y-2">
					<div className="text-xs text-muted-foreground">Target</div>
					<Input
						value={target}
						onChange={(e) =>
							setSelection({ manifestPath, target: e.target.value })
						}
						placeholder="ui_demo_train"
					/>
				</div>

				<div className="text-xs">
					<span className="text-muted-foreground">Status:</span>{" "}
					<span className="font-medium">{status}</span>
					{serverStatus && (
						<span className="text-muted-foreground">
							{" "}
							• server: <span className="font-medium">{serverStatus}</span>
						</span>
					)}
				</div>

				{run && (
					<div className="text-xs space-y-1">
						<div>
							<span className="text-muted-foreground">run_id:</span>{" "}
							<code className="text-[11px]">{run.id}</code>
						</div>
						<div>
							<span className="text-muted-foreground">pid:</span> {run.pid}
						</div>
					</div>
				)}

				{lastEvent.type && (
					<div className="text-xs space-y-1">
						<div>
							<span className="text-muted-foreground">event:</span>{" "}
							<span className="font-medium">{lastEvent.type}</span>
						</div>
						<div>
							<span className="text-muted-foreground">phase:</span>{" "}
							{lastEvent.phase ?? "—"}
						</div>
						<div>
							<span className="text-muted-foreground">step:</span>{" "}
							{lastEvent.step ?? "—"}
						</div>
					</div>
				)}

				{topMetrics.length > 0 && (
					<div className="rounded-md border border-border p-2">
						<div className="text-xs text-muted-foreground mb-2">
							Latest metrics
						</div>
						<div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
							{topMetrics.map(([k, v]) => (
								<React.Fragment key={k}>
									<div className="text-muted-foreground">{k}</div>
									<div className="font-mono text-right">{v.toFixed(4)}</div>
								</React.Fragment>
							))}
						</div>
					</div>
				)}

				{error && (
					<div className="rounded-md border border-destructive/40 bg-destructive/10 px-2 py-1 text-xs text-destructive">
						{error}
					</div>
				)}

				<div className="pt-1">
					<Button
						variant="outline"
						size="sm"
						type="button"
						onClick={() => setShowLogs((v) => !v)}
					>
						{showLogs ? "Hide logs" : "Show logs"}
					</Button>
				</div>

				{showLogs && (
					<div className="rounded-md border border-border bg-background p-2">
						<div className="text-xs text-muted-foreground mb-2 flex items-center justify-between">
							<span>Live logs</span>
							<span className="font-mono">
								{logLastTs
									? `${Math.round((Date.now() - logLastTs) / 1000)}s ago`
									: "—"}
							</span>
						</div>
						<pre className="text-[11px] leading-snug max-h-56 overflow-auto whitespace-pre-wrap">
							{logLines.length > 0 ? logLines.join("\n") : "—"}
						</pre>
					</div>
				)}
			</CardContent>
			<CardFooter className="gap-2">
				<Button
					onClick={() => startRun({ manifestPath, target })}
					disabled={!canStart}
				>
					Start
				</Button>
				<Button variant="outline" onClick={stopRun} disabled={!canStop}>
					Stop
				</Button>
			</CardFooter>
		</Card>
	);
}
