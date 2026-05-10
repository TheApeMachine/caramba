import { createFileRoute } from "@tanstack/react-router";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8118";

export const Route = createFileRoute("/api/compute/optimizer")({
	server: {
		handlers: {
			GET: async () => {
				const res = await fetch(`${BACKEND}/backend/compute/optimizer`);
				const data = await res.json();
				return new Response(JSON.stringify(data), {
					status: res.status,
					headers: { "Content-Type": "application/json" },
				});
			},
		},
	},
});
