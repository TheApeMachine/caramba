import { createFileRoute } from "@tanstack/react-router";

import TransformerEducational from "@/components/network/stepped";
import { RunPanel } from "@/components/run-panel";
import { RunProvider } from "@/lib/run-context";

export const Route = createFileRoute("/stepped")({
	component: SteppedRoute,
});

function SteppedRoute() {
	return (
		<RunProvider>
			<div className="relative w-full h-screen">
				<TransformerEducational />
				<div className="absolute top-4 right-4 z-50 w-[360px] pointer-events-auto">
					<RunPanel />
				</div>
			</div>
		</RunProvider>
	);
}

