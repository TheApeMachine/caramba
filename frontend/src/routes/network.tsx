import { createFileRoute } from "@tanstack/react-router";

import NetworkDemo from "@/components/network/network";
import { RunPanel } from "@/components/run-panel";
import { RunProvider } from "@/lib/run-context";

export const Route = createFileRoute("/network")({
	component: NetworkRoute,
});

function NetworkRoute() {
	return (
		<RunProvider>
			<div className="relative w-full h-screen">
				<NetworkDemo />
				<div className="absolute top-4 right-4 z-50 w-[360px] pointer-events-auto">
					<RunPanel />
				</div>
			</div>
		</RunProvider>
	);
}

