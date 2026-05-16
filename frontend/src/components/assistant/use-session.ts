import { useCallback, useState } from "react";
import type { Persona, Session } from "./types";
import {
	deriveTitle,
	loadActiveId,
	loadSessions,
	newSession,
	saveActiveId,
	saveSessions,
} from "./storage";

function init(): { sessions: Session[]; activeId: string } {
	const sessions = loadSessions();
	const storedId = loadActiveId();

	if (sessions.length === 0) {
		const fresh = newSession();
		saveSessions([fresh]);
		saveActiveId(fresh.id);
		return { sessions: [fresh], activeId: fresh.id };
	}

	const activeId = sessions.find((s) => s.id === storedId)
		? (storedId as string)
		: sessions[0].id;

	return { sessions, activeId };
}

export function useSession() {
	const [{ sessions, activeId }, setState] = useState(init);

	const session = sessions.find((s) => s.id === activeId) ?? sessions[0];

	const persist = useCallback((next: Session[]) => {
		saveSessions(next);
		setState((prev) => ({ ...prev, sessions: next }));
	}, []);

	const setActive = useCallback((id: string) => {
		saveActiveId(id);
		setState((prev) => ({ ...prev, activeId: id }));
	}, []);

	const createSession = useCallback(() => {
		const fresh = newSession();
		setState((prev) => {
			const next = [fresh, ...prev.sessions];
			saveSessions(next);
			saveActiveId(fresh.id);
			return { sessions: next, activeId: fresh.id };
		});
	}, []);

	const deleteSession = useCallback((id: string) => {
		setState((prev) => {
			const next = prev.sessions.filter((s) => s.id !== id);
			if (next.length === 0) {
				const fresh = newSession();
				saveSessions([fresh]);
				saveActiveId(fresh.id);
				return { sessions: [fresh], activeId: fresh.id };
			}
			const activeId = prev.activeId === id ? next[0].id : prev.activeId;
			saveSessions(next);
			saveActiveId(activeId);
			return { sessions: next, activeId };
		});
	}, []);

	const appendMessages = useCallback(
		(incoming: Session["messages"]) => {
			setState((prev) => {
				const next = prev.sessions.map((s) => {
					if (s.id !== prev.activeId) return s;
					const merged = [...s.messages, ...incoming];
					const title = s.messages.length === 0 ? deriveTitle({ ...s, messages: merged }) : s.title;
					return { ...s, messages: merged, title };
				});
				saveSessions(next);
				return { ...prev, sessions: next };
			});
		},
		[],
	);

	const upsertMessage = useCallback(
		(message: Session["messages"][number]) => {
			setState((prev) => {
				const next = prev.sessions.map((s) => {
					if (s.id !== prev.activeId) return s;
					const idx = s.messages.findIndex((m) => m.id === message.id);
					const merged =
						idx === -1
							? [...s.messages, message]
							: s.messages.map((m, i) => (i === idx ? message : m));
					return { ...s, messages: merged };
				});
				saveSessions(next);
				return { ...prev, sessions: next };
			});
		},
		[],
	);

	const updatePersona = useCallback((persona: Persona) => {
		setState((prev) => {
			const next = prev.sessions.map((s) => {
				if (s.id !== prev.activeId) return s;
				const personas = s.personas.map((p) => (p.id === persona.id ? persona : p));
				return { ...s, personas };
			});
			saveSessions(next);
			return { ...prev, sessions: next };
		});
	}, []);

	const addPersona = useCallback((persona: Persona) => {
		setState((prev) => {
			const next = prev.sessions.map((s) => {
				if (s.id !== prev.activeId) return s;
				return { ...s, personas: [...s.personas, persona] };
			});
			saveSessions(next);
			return { ...prev, sessions: next };
		});
	}, []);

	const removePersona = useCallback((personaId: string) => {
		setState((prev) => {
			const next = prev.sessions.map((s) => {
				if (s.id !== prev.activeId) return s;
				const personas = s.personas.filter((p) => p.id !== personaId);
				return { ...s, personas: personas.length > 0 ? personas : s.personas };
			});
			saveSessions(next);
			return { ...prev, sessions: next };
		});
	}, []);

	const setWindowSize = useCallback((size: number) => {
		setState((prev) => {
			const next = prev.sessions.map((s) =>
				s.id !== prev.activeId ? s : { ...s, windowSize: size },
			);
			saveSessions(next);
			return { ...prev, sessions: next };
		});
	}, []);

	return {
		sessions,
		session,
		setActive,
		createSession,
		deleteSession,
		appendMessages,
		upsertMessage,
		updatePersona,
		addPersona,
		removePersona,
		setWindowSize,
		persist,
	};
}
