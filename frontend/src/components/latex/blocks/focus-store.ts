"use client";

import { Store, useSelector } from "@tanstack/react-store";

type PaperEditorFocus = {
	focusedBlockId: string | null;
};

/*
paperEditorFocusStore holds the editor's ephemeral UI focus state. It is
module-scoped because at any moment there is exactly one paper editor
mounted; if that assumption changes, this becomes a per-instance store
created in the provider.
*/
const paperEditorFocusStore = new Store<PaperEditorFocus>({
	focusedBlockId: null,
});

export const useFocusedBlockId = (): string | null =>
	useSelector(paperEditorFocusStore, (state) => state.focusedBlockId);

export const setFocusedBlockId = (id: string | null): void => {
	paperEditorFocusStore.setState((previous) => {
		if (previous.focusedBlockId === id) {
			return previous;
		}

		return { focusedBlockId: id };
	});
};
