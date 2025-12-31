import { Link, createFileRoute } from "@tanstack/react-router";

import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({ component: App });

function App() {
	return (
		<div className="min-h-screen bg-background text-foreground">
			<div className="mx-auto max-w-3xl p-6 space-y-6">
				<div className="space-y-1">
					<div className="text-2xl font-semibold">caramba</div>
					<div className="text-sm text-muted-foreground">
						UI demos wired to real training/inference plumbing.
					</div>
				</div>

				<div className="flex flex-wrap gap-2">
					<Link to="/network" className={cn(buttonVariants({ variant: "default" }))}>
						Network demo
					</Link>
					<Link to="/stepped" className={cn(buttonVariants({ variant: "outline" }))}>
						Stepped demo
					</Link>
				</div>

				<div className="text-sm text-muted-foreground">
					Start the backend with <code className="font-mono">caramba serve</code> and
					then click a demo.
				</div>
			</div>
		</div>
	);
}