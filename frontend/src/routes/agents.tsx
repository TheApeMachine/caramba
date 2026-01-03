import { createFileRoute, Link } from "@tanstack/react-router";

import { AgentPanel } from "@/components/agent-panel";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/agents")({ component: AgentsPage });

function AgentsPage() {
	return (
		<div className="min-h-screen bg-background text-foreground">
			<div className="mx-auto max-w-5xl p-6 space-y-6">
				<div className="flex items-center justify-between">
					<div className="space-y-1">
						<div className="text-2xl font-semibold flex items-center gap-2">
							<span>🤖</span>
							<span>Agent System</span>
						</div>
						<div className="text-sm text-muted-foreground">
							Configure and run multi-agent workflows
						</div>
					</div>
					<Link
						to="/"
						className={cn(buttonVariants({ variant: "outline", size: "sm" }))}
					>
						← Back
					</Link>
				</div>

				<AgentPanel />

				<div className="text-xs text-muted-foreground space-y-1">
					<p>
						<strong>Note:</strong> Agents require the backend to be running with{" "}
						<code className="font-mono bg-muted px-1 rounded">
							caramba serve
						</code>
					</p>
					<p>
						Available processes: Discussion, Paper Write, Paper Review, Research
						Loop, Code Graph Sync, Platform Improve
					</p>
				</div>
			</div>
		</div>
	);
}
