/*
Throwaway Jazz 2.0 round-trip demo at /jazz-test.

Self-contained: it mounts its OWN local-first JazzProvider (no Clerk / JWKS
needed), so it can't affect the rest of the app. Open /jazz-test in two browser
tabs, click "Add project" in one, and watch it appear in the other — that's the
single-WebSocket live sync working end to end.

Delete this file once Phase 0 is proven.

APIs verified against jazz-tools@2.0.0-alpha.50:
  useLocalFirstAuth() -> { secret, isLoading }
  <JazzProvider config={{ appId, serverUrl, secret }} createJazzClient={createJazzClient}>
  useDb() / useSession() (session.user_id) / useAll(query)
  db.insert(app.table, values) -> { value }
*/

import { createFileRoute } from "@tanstack/react-router";
import {
	createJazzClient,
	JazzProvider,
	useDb,
	useLocalFirstAuth,
	useSession,
	useAll,
} from "jazz-tools/react";
import { app } from "../../schema";

export const Route = createFileRoute("/jazz-test")({
	component: JazzTestPage,
});

const APP_ID = import.meta.env.VITE_JAZZ_APP_ID as string;
const SERVER_URL = import.meta.env.VITE_JAZZ_SERVER_URL as string;

function JazzTestPage() {
	const { secret, isLoading } = useLocalFirstAuth();

	if (isLoading || !secret) {
		return <p style={{ padding: 24 }}>Starting local Jazz identity…</p>;
	}

	return (
		<JazzProvider
			config={{ appId: APP_ID, serverUrl: SERVER_URL, secret }}
			createJazzClient={createJazzClient}
		>
			<Demo />
		</JazzProvider>
	);
}

function Demo() {
	const db = useDb();
	const session = useSession();
	const projects = useAll(app.projects.where({}));

	const addProject = () => {
		if (!session) {
			return;
		}

		const now = new Date();
		const { value: project } = db.insert(app.projects, {
			name: `Demo project ${now.toLocaleTimeString()}`,
			description: "",
			organization_slug: "demo",
			created_at: now,
			updated_at: now,
		});

		// A read of projects requires a matching projectMembers row (see
		// permissions.ts), so add the current user as an owner of what we just made.
		db.insert(app.projectMembers, {
			project: project.id,
			user_id: session.user_id,
			role: "owner",
			created_at: now,
		});
	};

	return (
		<div style={{ padding: 24, fontFamily: "monospace" }}>
			<h1>Jazz round-trip test</h1>
			<p>Signed-in Jazz user: {session?.user_id ?? "(none)"}</p>
			<button type="button" onClick={addProject}>
				Add project
			</button>
			<p>Open this page in a second tab — new rows appear in both instantly.</p>
			<ul>
				{(projects ?? []).map((project) => (
					<li key={project.id}>
						{project.name} · {project.organization_slug}
					</li>
				))}
			</ul>
			{projects === undefined ? <p>Loading projects…</p> : null}
		</div>
	);
}
