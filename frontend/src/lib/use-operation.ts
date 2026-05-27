"use client";

import { Store, useSelector } from "@tanstack/react-store";
import { useRef } from "react";

export type OperationStatus = "idle" | "pending" | "error";

type OperationState = {
	status: OperationStatus;
	error: string | null;
};

export type OperationHandle = {
	status: OperationStatus;
	error: string | null;
	isPending: boolean;
	run: (action: () => Promise<void> | void) => Promise<void>;
	reset: () => void;
};

/*
useOperation owns the idle / pending / error lifecycle of a single
imperative async action (e.g. a "Create…" button) in a per-instance
Tanstack Store. Subscribers re-render through useStore selectors so
the component can flip its label / disabled state without ever
reaching for useState.
*/
export const useOperation = (): OperationHandle => {
	const storeRef = useRef<Store<OperationState> | null>(null);

	if (storeRef.current === null) {
		storeRef.current = new Store<OperationState>({
			status: "idle",
			error: null,
		});
	}

	const store = storeRef.current;
	const state = useSelector(store, (current) => current);

	const run = async (action: () => Promise<void> | void): Promise<void> => {
		store.setState(() => ({ status: "pending", error: null }));

		try {
			await action();
			store.setState(() => ({ status: "idle", error: null }));
		} catch (cause) {
			const message = cause instanceof Error ? cause.message : String(cause);
			store.setState(() => ({ status: "error", error: message }));
		}
	};

	const reset = (): void => {
		store.setState(() => ({ status: "idle", error: null }));
	};

	return {
		status: state.status,
		error: state.error,
		isPending: state.status === "pending",
		run,
		reset,
	};
};
