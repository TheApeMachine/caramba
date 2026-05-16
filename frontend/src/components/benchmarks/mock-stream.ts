import { useEffect, useMemo, useRef, useState } from "react";
import { type BenchmarkSpec, DATASETS } from "./model";

export type RunStatus = "queued" | "running" | "done" | "failed";

export interface RunStepSample {
	step: number;
	elapsedSeconds: number;
	loss: number;
	accuracy: number;
	throughput: number;
	latencyMs: number;
}

export interface RunEvent {
	timestamp: number;
	level: "info" | "warn" | "error";
	message: string;
}

export interface RunState {
	status: RunStatus;
	startedAt: number;
	steps: RunStepSample[];
	events: RunEvent[];
	confusion: number[][];
	classLabels: string[];
	latencies: number[];
	totalSamples: number;
	processedSamples: number;
	eta: number;
}

const hashSeed = (input: string): number => {
	let hash = 0;
	for (let i = 0; i < input.length; i++) {
		hash = (hash << 5) - hash + input.charCodeAt(i);
		hash |= 0;
	}
	return Math.abs(hash) || 1;
};

const mulberry = (seed: number): (() => number) => {
	let state = seed >>> 0;
	return () => {
		state = (state + 0x6d2b79f5) >>> 0;
		let t = state;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
};

const makeStartEvents = (spec: BenchmarkSpec): RunEvent[] => {
	const now = Date.now();
	return [
		{
			timestamp: now,
			level: "info",
			message: `Run "${spec.name || "untitled"}" queued on ${spec.backend}.`,
		},
		{
			timestamp: now + 250,
			level: "info",
			message: `Loading checkpoint ${spec.modelId}.`,
		},
		{
			timestamp: now + 600,
			level: "info",
			message: `Streaming dataset ${spec.datasetId} (batch=${spec.batchSize}).`,
		},
		{
			timestamp: now + 900,
			level: "info",
			message: "Warm-up complete. Beginning evaluation.",
		},
	];
};

const buildConfusion = (
	classes: string[],
	rng: () => number,
	accuracy: number,
): number[][] => {
	const n = classes.length;
	const matrix: number[][] = Array.from({ length: n }, () =>
		Array.from({ length: n }, () => 0),
	);
	const perClass = 50;
	for (let actual = 0; actual < n; actual++) {
		for (let i = 0; i < perClass; i++) {
			const hit = rng() < accuracy;
			const predicted = hit ? actual : Math.floor(rng() * n);
			matrix[actual][predicted] += 1;
		}
	}
	return matrix;
};

const STEP_INTERVAL_MS = 500;
const STEPS_PER_RUN = 60;

/*
useMockRun simulates a streaming benchmark run. Output shape matches what a
real eval loop would push: per-step quality metrics, a rolling latency
window, periodic event log lines, and a confusion matrix that refines over
time. The hook is stable: changing spec resets the run.
*/
export const useMockRun = (
	spec: BenchmarkSpec,
	options: { autoStart?: boolean; runId?: string } = {},
): RunState & { restart: () => void; stop: () => void } => {
	const { autoStart = true, runId } = options;
	const seedKey = `${runId ?? spec.name}-${spec.modelId}-${spec.datasetId}-${spec.backend}`;
	const rngRef = useRef<() => number>(mulberry(hashSeed(seedKey)));

	const dataset = useMemo(
		() => DATASETS.find((entry) => entry.id === spec.datasetId),
		[spec.datasetId],
	);
	const classes = dataset?.classes ?? ["A", "B"];
	const cappedClasses = classes.length > 8 ? classes.slice(0, 8) : classes;

	const [state, setState] = useState<RunState>(() => ({
		status: autoStart ? "running" : "queued",
		startedAt: Date.now(),
		steps: [],
		events: makeStartEvents(spec),
		confusion: cappedClasses.map((_) => cappedClasses.map(() => 0)),
		classLabels: cappedClasses,
		latencies: [],
		totalSamples: dataset?.size ?? 1000,
		processedSamples: 0,
		eta: 0,
	}));

	const tickRef = useRef(0);
	const intervalRef = useRef<number | null>(null);

	const restart = () => {
		tickRef.current = 0;
		rngRef.current = mulberry(hashSeed(`${seedKey}-${Date.now()}`));
		setState({
			status: "running",
			startedAt: Date.now(),
			steps: [],
			events: makeStartEvents(spec),
			confusion: cappedClasses.map((_) => cappedClasses.map(() => 0)),
			classLabels: cappedClasses,
			latencies: [],
			totalSamples: dataset?.size ?? 1000,
			processedSamples: 0,
			eta: 0,
		});
	};

	const stop = () => {
		if (intervalRef.current !== null) {
			window.clearInterval(intervalRef.current);
			intervalRef.current = null;
		}
		setState((current) =>
			current.status === "running" ? { ...current, status: "done" } : current,
		);
	};

	const restartRef = useRef(restart);
	restartRef.current = restart;

	useEffect(() => {
		void seedKey;
		restartRef.current();
	}, [seedKey]);

	useEffect(() => {
		if (state.status !== "running") return;

		intervalRef.current = window.setInterval(() => {
			tickRef.current += 1;
			const step = tickRef.current;
			const rng = rngRef.current;

			setState((current) => {
				const progress = Math.min(1, step / STEPS_PER_RUN);
				const noise = (rng() - 0.5) * 0.04;
				const accuracy = Math.min(
					0.99,
					0.4 + progress * 0.45 + Math.sin(step / 6) * 0.02 + noise,
				);
				const loss = Math.max(
					0.05,
					2.0 - progress * 1.5 + (rng() - 0.5) * 0.08,
				);
				const throughput = 800 + Math.sin(step / 4) * 40 + rng() * 30;
				const latencyMs = 1200 / throughput + (rng() - 0.5) * 0.3;

				const sample: RunStepSample = {
					step,
					elapsedSeconds: (Date.now() - current.startedAt) / 1000,
					loss: Number(loss.toFixed(4)),
					accuracy: Number(accuracy.toFixed(4)),
					throughput: Number(throughput.toFixed(1)),
					latencyMs: Number(latencyMs.toFixed(2)),
				};

				const steps = [...current.steps, sample];
				const latencies = [
					...current.latencies,
					...Array.from({ length: 8 }, () => latencyMs + (rng() - 0.5) * 0.4),
				].slice(-512);

				const processedSamples = Math.min(
					current.totalSamples,
					Math.round(progress * current.totalSamples),
				);
				const eta = ((1 - progress) * STEP_INTERVAL_MS * STEPS_PER_RUN) / 1000;

				const events = [...current.events];
				if (step === 1)
					events.push({
						timestamp: Date.now(),
						level: "info",
						message: `Step 1 complete. First batch through the pipeline.`,
					});
				if (step === Math.round(STEPS_PER_RUN / 2))
					events.push({
						timestamp: Date.now(),
						level: "info",
						message: "Halfway through evaluation set.",
					});
				if (step % 15 === 0)
					events.push({
						timestamp: Date.now(),
						level: "info",
						message: `Step ${step}: acc=${accuracy.toFixed(3)} loss=${loss.toFixed(3)}.`,
					});
				if (step === STEPS_PER_RUN)
					events.push({
						timestamp: Date.now(),
						level: "info",
						message: `Run complete. Final accuracy ${accuracy.toFixed(3)}.`,
					});

				const confusion = buildConfusion(current.classLabels, rng, accuracy);

				const nextStatus: RunStatus =
					step >= STEPS_PER_RUN ? "done" : current.status;

				return {
					...current,
					status: nextStatus,
					steps,
					events,
					confusion,
					latencies,
					processedSamples,
					eta,
				};
			});
		}, STEP_INTERVAL_MS);

		return () => {
			if (intervalRef.current !== null) {
				window.clearInterval(intervalRef.current);
				intervalRef.current = null;
			}
		};
	}, [state.status]);

	return { ...state, restart, stop };
};
