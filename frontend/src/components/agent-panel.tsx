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
import { Textarea } from "@/components/ui/textarea";
import { apiUrl } from "@/lib/api";

// Process types available in the agent system
type ProcessType =
	| "discussion"
	| "paper_write"
	| "paper_review"
	| "research_loop"
	| "code_graph_sync"
	| "platform_improve";

type AgentStatus = "idle" | "loading" | "running" | "completed" | "error";

type PersonaInfo = {
	name: string;
	model: string;
	temperature: number;
	mcp_servers: string[];
};

type ProcessConfig = {
	type: ProcessType;
	name: string;
	topic?: string;
	leader?: string;
	writer?: string;
	reviewer?: string;
	max_rounds?: number;
	max_iterations?: number;
};

type AgentMessage = {
	id: string;
	agent: string;
	role: "user" | "assistant" | "system";
	content: string;
	timestamp: number;
};

type AgentRun = {
	id: string;
	process_type: ProcessType;
	status: AgentStatus;
	started_at: number;
	ended_at?: number;
	messages: AgentMessage[];
	result?: unknown;
	error?: string;
};

export function AgentPanel() {
	// Process selection and configuration
	const [processType, setProcessType] = React.useState<ProcessType>("discussion");
	const [topic, setTopic] = React.useState("");
	const [manifestPath, setManifestPath] = React.useState("config/presets/agents.yml");
	
	// Team configuration
	const [teamConfig, setTeamConfig] = React.useState<Record<string, string>>({
		leader: "researcher",
		writer: "writer",
		reviewer: "critic",
	});
	
	// Available personas (loaded from backend)
	const [personas, setPersonas] = React.useState<PersonaInfo[]>([]);
	const [personasLoading, setPersonasLoading] = React.useState(false);
	
	// Run state
	const [status, setStatus] = React.useState<AgentStatus>("idle");
	const [currentRun, setCurrentRun] = React.useState<AgentRun | null>(null);
	const [messages, setMessages] = React.useState<AgentMessage[]>([]);
	const [error, setError] = React.useState<string | null>(null);
	
	// Streaming state
	const [streamText, setStreamText] = React.useState("");
	const eventSourceRef = React.useRef<EventSource | null>(null);
	const messagesEndRef = React.useRef<HTMLDivElement>(null);

	// Scroll to bottom when new messages arrive
	React.useEffect(() => {
		messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
	});

	const loadPersonas = React.useCallback(async () => {
		setPersonasLoading(true);
		try {
			const res = await fetch(apiUrl("/api/agents/personas"));
			if (res.ok) {
				const data = await res.json();
				if (Array.isArray(data.personas)) {
					setPersonas(data.personas);
				}
			}
		} catch (e) {
			console.error("Failed to load personas:", e);
		} finally {
			setPersonasLoading(false);
		}
	}, []);

	// Load available personas on mount
	React.useEffect(() => {
		loadPersonas();
	}, [loadPersonas]);

	const startProcess = async () => {
		setStatus("loading");
		setError(null);
		setMessages([]);
		setStreamText("");

		const processConfig: ProcessConfig = {
			type: processType,
			name: `${processType}_${Date.now()}`,
			topic,
		};

		// Add process-specific config
		if (processType === "discussion") {
			processConfig.leader = teamConfig.leader;
			processConfig.max_rounds = 5;
		} else if (processType === "paper_write") {
			processConfig.writer = teamConfig.writer;
		} else if (processType === "paper_review") {
			processConfig.reviewer = teamConfig.reviewer;
		} else if (processType === "research_loop") {
			processConfig.leader = teamConfig.leader;
			processConfig.writer = teamConfig.writer;
			processConfig.reviewer = teamConfig.reviewer;
			processConfig.max_iterations = 3;
		}

		try {
			const res = await fetch(apiUrl("/api/agents/run"), {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					manifest_path: manifestPath,
					process: processConfig,
					team: teamConfig,
				}),
			});

			if (!res.ok) {
				const errData = await res.json().catch(() => ({}));
				throw new Error(errData.error || `HTTP ${res.status}`);
			}

			const data = await res.json();
			const runId = data.run_id;

			setCurrentRun({
				id: runId,
				process_type: processType,
				status: "running",
				started_at: Date.now(),
				messages: [],
			});
			setStatus("running");

			// Connect to SSE stream for live updates
			connectToStream(runId);
		} catch (e) {
			setError(e instanceof Error ? e.message : "Failed to start process");
			setStatus("error");
		}
	};

	const connectToStream = (runId: string) => {
		// Close existing connection
		if (eventSourceRef.current) {
			eventSourceRef.current.close();
		}

		const es = new EventSource(apiUrl(`/api/agents/stream/${runId}`));
		eventSourceRef.current = es;

		es.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				handleStreamEvent(data);
			} catch (e) {
				console.error("Failed to parse stream event:", e);
			}
		};

		es.onerror = () => {
			es.close();
			eventSourceRef.current = null;
			// Don't set error if we completed successfully
			if (status === "running") {
				setStatus("completed");
			}
		};

		es.addEventListener("message", (event) => {
			try {
				const data = JSON.parse(event.data);
				handleStreamEvent(data);
			} catch (e) {
				console.error("Failed to parse message event:", e);
			}
		});

		es.addEventListener("done", () => {
			es.close();
			eventSourceRef.current = null;
			setStatus("completed");
		});

		es.addEventListener("error", (event) => {
			try {
				const data = JSON.parse((event as MessageEvent).data);
				setError(data.error || "Process failed");
				setStatus("error");
			} catch {
				// Generic error
			}
			es.close();
			eventSourceRef.current = null;
		});
	};

	const handleStreamEvent = (data: {
		type: string;
		agent?: string;
		content?: string;
		delta?: string;
		result?: unknown;
	}) => {
		switch (data.type) {
			case "agent_start":
				// An agent started processing
				setMessages((prev) => [
					...prev,
					{
						id: `${Date.now()}_start`,
						agent: data.agent || "system",
						role: "system",
						content: `${data.agent} started...`,
						timestamp: Date.now(),
					},
				]);
				break;

			case "reasoning_delta":
				// Streaming reasoning text
				setStreamText((prev) => prev + (data.delta || ""));
				break;

			case "output_delta":
				// Streaming output text
				setStreamText((prev) => prev + (data.delta || ""));
				break;

			case "agent_complete":
				// Agent finished - save accumulated stream text as message
				if (streamText.trim()) {
					setMessages((prev) => [
						...prev,
						{
							id: `${Date.now()}_msg`,
							agent: data.agent || "agent",
							role: "assistant",
							content: streamText,
							timestamp: Date.now(),
						},
					]);
				}
				setStreamText("");
				break;

			case "message":
				// Complete message
				setMessages((prev) => [
					...prev,
					{
						id: `${Date.now()}_msg`,
						agent: data.agent || "agent",
						role: "assistant",
						content: data.content || "",
						timestamp: Date.now(),
					},
				]);
				break;

			case "result":
				// Final result
				setCurrentRun((prev) =>
					prev
						? {
								...prev,
								result: data.result,
								status: "completed",
								ended_at: Date.now(),
							}
						: null,
				);
				break;

			case "error":
				setError(data.content || "Unknown error");
				setStatus("error");
				break;
		}
	};

	const stopProcess = async () => {
		if (!currentRun) return;

		try {
			await fetch(apiUrl(`/api/agents/stop/${currentRun.id}`), {
				method: "POST",
			});
		} catch (e) {
			console.error("Failed to stop process:", e);
		}

		// Close stream
		if (eventSourceRef.current) {
			eventSourceRef.current.close();
			eventSourceRef.current = null;
		}

		setStatus("idle");
	};

	const processDescriptions: Record<ProcessType, string> = {
		discussion: "Multi-agent discussion with a leader coordinating the conversation",
		paper_write: "Write a research paper or document with an AI writer",
		paper_review: "Review and critique a paper with suggested improvements",
		research_loop: "Iterative research cycle: discuss → write → review → refine",
		code_graph_sync: "Sync codebase to knowledge graph for agent understanding",
		platform_improve: "Full platform improvement workflow with multiple agents",
	};

	const canStart = status === "idle" || status === "completed" || status === "error";
	const canStop = status === "running" || status === "loading";

	return (
		<Card className="w-full max-w-4xl">
			<CardHeader>
				<CardTitle className="flex items-center gap-2">
					<span className="text-2xl">🤖</span>
					Agent Control Panel
				</CardTitle>
			</CardHeader>
			<CardContent className="space-y-6">
				{/* Process Type Selection */}
				<div className="space-y-3">
					<div className="text-sm font-medium text-foreground">Process Type</div>
					<div className="grid grid-cols-2 md:grid-cols-3 gap-2">
						{(
							[
								"discussion",
								"paper_write",
								"paper_review",
								"research_loop",
								"code_graph_sync",
								"platform_improve",
							] as ProcessType[]
						).map((type) => (
							<button
								key={type}
								type="button"
								onClick={() => setProcessType(type)}
								className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
									processType === type
										? "bg-primary text-primary-foreground"
										: "bg-muted hover:bg-muted/80 text-muted-foreground"
								}`}
							>
								{type.replace(/_/g, " ")}
							</button>
						))}
					</div>
					<p className="text-xs text-muted-foreground">
						{processDescriptions[processType]}
					</p>
				</div>

				{/* Topic/Goal Input */}
				<div className="space-y-2">
					<div className="text-sm font-medium text-foreground">
						{processType === "discussion" ? "Discussion Topic" : "Goal / Task"}
					</div>
					<Textarea
						value={topic}
						onChange={(e) => setTopic(e.target.value)}
						placeholder={
							processType === "discussion"
								? "What should the agents discuss? e.g., 'Best practices for neural network optimization'"
								: "What should the agent accomplish? e.g., 'Write a technical blog post about transformers'"
						}
						rows={3}
					/>
				</div>

				{/* Manifest Path */}
				<div className="space-y-2">
					<div className="text-sm font-medium text-foreground">
						Agent Manifest Path
					</div>
					<Input
						value={manifestPath}
						onChange={(e) => setManifestPath(e.target.value)}
						placeholder="config/presets/agents.yml"
					/>
				</div>

				{/* Team Configuration */}
				<div className="space-y-3">
					<div className="text-sm font-medium text-foreground">Team Configuration</div>
					<div className="grid grid-cols-1 md:grid-cols-3 gap-3">
						{processType === "discussion" && (
							<div className="space-y-1">
								<span className="text-xs text-muted-foreground">Leader</span>
								<Input
									value={teamConfig.leader || ""}
									onChange={(e) =>
										setTeamConfig((prev) => ({ ...prev, leader: e.target.value }))
									}
									placeholder="researcher"
									aria-label="Leader persona name"
								/>
							</div>
						)}
						{(processType === "paper_write" || processType === "research_loop") && (
							<div className="space-y-1">
								<span className="text-xs text-muted-foreground">Writer</span>
								<Input
									value={teamConfig.writer || ""}
									onChange={(e) =>
										setTeamConfig((prev) => ({ ...prev, writer: e.target.value }))
									}
									placeholder="writer"
									aria-label="Writer persona name"
								/>
							</div>
						)}
						{(processType === "paper_review" || processType === "research_loop") && (
							<div className="space-y-1">
								<span className="text-xs text-muted-foreground">Reviewer</span>
								<Input
									value={teamConfig.reviewer || ""}
									onChange={(e) =>
										setTeamConfig((prev) => ({ ...prev, reviewer: e.target.value }))
									}
									placeholder="critic"
									aria-label="Reviewer persona name"
								/>
							</div>
						)}
						{processType === "research_loop" && (
							<div className="space-y-1">
								<span className="text-xs text-muted-foreground">Leader</span>
								<Input
									value={teamConfig.leader || ""}
									onChange={(e) =>
										setTeamConfig((prev) => ({ ...prev, leader: e.target.value }))
									}
									placeholder="researcher"
									aria-label="Leader persona name for research loop"
								/>
							</div>
						)}
					</div>
					{personas.length > 0 && (
						<div className="text-xs text-muted-foreground">
							Available personas:{" "}
							{personas.map((p) => p.name).join(", ")}
						</div>
					)}
					{personasLoading && (
						<div className="text-xs text-muted-foreground">Loading personas...</div>
					)}
				</div>

				{/* Status Display */}
				<div className="flex items-center gap-3 py-2">
					<div
						className={`w-3 h-3 rounded-full ${
							status === "idle"
								? "bg-gray-400"
								: status === "loading"
									? "bg-yellow-400 animate-pulse"
									: status === "running"
										? "bg-green-400 animate-pulse"
										: status === "completed"
											? "bg-blue-400"
											: "bg-red-400"
						}`}
					/>
					<span className="text-sm font-medium capitalize">{status}</span>
					{currentRun && (
						<span className="text-xs text-muted-foreground">
							Run ID: {currentRun.id.slice(0, 8)}...
						</span>
					)}
				</div>

				{/* Error Display */}
				{error && (
					<div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
						{error}
					</div>
				)}

				{/* Messages / Conversation View */}
				{(messages.length > 0 || streamText) && (
					<div className="space-y-2">
						<div className="text-sm font-medium text-foreground">
							Agent Conversation
						</div>
						<div className="rounded-md border border-border bg-background/50 max-h-96 overflow-y-auto">
							<div className="p-3 space-y-3">
								{messages.map((msg) => (
									<div
										key={msg.id}
										className={`flex flex-col gap-1 ${
											msg.role === "system" ? "opacity-60" : ""
										}`}
									>
										<div className="flex items-center gap-2">
											<span
												className={`text-xs font-medium px-2 py-0.5 rounded ${
													msg.role === "system"
														? "bg-gray-500/20 text-gray-400"
														: "bg-primary/20 text-primary"
												}`}
											>
												{msg.agent}
											</span>
											<span className="text-xs text-muted-foreground">
												{new Date(msg.timestamp).toLocaleTimeString()}
											</span>
										</div>
										<div className="text-sm pl-2 border-l-2 border-border whitespace-pre-wrap">
											{msg.content}
										</div>
									</div>
								))}
								{/* Streaming text */}
								{streamText && (
									<div className="flex flex-col gap-1">
										<div className="flex items-center gap-2">
											<span className="text-xs font-medium px-2 py-0.5 rounded bg-yellow-500/20 text-yellow-500">
												streaming...
											</span>
										</div>
										<div className="text-sm pl-2 border-l-2 border-yellow-500/50 whitespace-pre-wrap">
											{streamText}
											<span className="inline-block w-2 h-4 bg-yellow-500 animate-pulse ml-0.5" />
										</div>
									</div>
								)}
								<div ref={messagesEndRef} />
							</div>
						</div>
					</div>
				)}

				{/* Result Display */}
				{currentRun?.result && (
					<div className="space-y-2">
						<div className="text-sm font-medium text-foreground">Result</div>
						<pre className="rounded-md border border-border bg-background/50 p-3 text-xs overflow-auto max-h-48">
							{JSON.stringify(currentRun.result, null, 2)}
						</pre>
					</div>
				)}
			</CardContent>
			<CardFooter className="gap-2">
				<Button
					onClick={startProcess}
					disabled={!canStart || !topic.trim()}
				>
					{status === "loading" ? "Starting..." : "Start Process"}
				</Button>
				<Button
					variant="outline"
					onClick={stopProcess}
					disabled={!canStop}
				>
					Stop
				</Button>
				<Button
					variant="ghost"
					onClick={() => {
						setMessages([]);
						setStreamText("");
						setError(null);
						setCurrentRun(null);
						setStatus("idle");
					}}
				>
					Clear
				</Button>
			</CardFooter>
		</Card>
	);
}
