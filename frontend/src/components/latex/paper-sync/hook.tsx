"use client";

import { useSelector } from "@tanstack/react-store";
import type { Dispatch } from "react";
import { useEffect, useRef } from "react";
import {
	createPaperSyncController,
	type PaperSyncController,
} from "#/components/latex/paper-sync/controller";
import type { PaperAction } from "#/components/latex/model/paper-reducer";
import type { PaperBlock, PaperMetadata } from "#/components/latex/model/types";

type MetadataWriter = {
	setFieldValue: (name: keyof PaperMetadata, value: string) => void;
};

export type UseResearchPaperCollectionSyncArgs = {
	paperIdProp?: string;
	bootstrapProjectId?: string;
	onPaperBootstrapped?: (paperId: string) => void;
	dispatch: Dispatch<PaperAction>;
	blocksRef: { current: PaperBlock[] };
	blocks: PaperBlock[];
	metadata: PaperMetadata;
	metadataForm: MetadataWriter;
};

export type ResearchPaperSyncResult = {
	effectivePaperId: string | null;
	persistEnabled: boolean;
	ready: boolean;
	waitingForRemote: boolean;
	bootstrapError: string | null;
	saveError: string | null;
};

/*
useResearchPaperCollectionSync is a thin React wrapper over
PaperSyncController. The controller owns every piece of sync state
in a Tanstack Store + collection subscription, so this hook only
threads props through, subscribes to the controller's state, and
disposes the controller on unmount.
*/
export const useResearchPaperCollectionSync = (
	args: UseResearchPaperCollectionSyncArgs,
): ResearchPaperSyncResult => {
	const controllerRef = useRef<PaperSyncController | null>(null);

	if (controllerRef.current === null) {
		controllerRef.current = createPaperSyncController({
			paperIdProp: args.paperIdProp,
			bootstrapProjectId: args.bootstrapProjectId,
			onPaperBootstrapped: args.onPaperBootstrapped,
			getDocument: () => ({
				blocks: args.blocksRef.current,
				metadata: args.metadata,
			}),
			applyDocument: ({ blocks, metadata }) => {
				args.dispatch({ type: "REPLACE_BLOCKS", blocks });
				args.metadataForm.setFieldValue("title", metadata.title);
				args.metadataForm.setFieldValue("authors", metadata.authors);
				args.metadataForm.setFieldValue("keywords", metadata.keywords);
				args.metadataForm.setFieldValue("abstract", metadata.abstract);
			},
		});
	}

	const controller = controllerRef.current;

	controller.configure({
		paperIdProp: args.paperIdProp,
		bootstrapProjectId: args.bootstrapProjectId,
		onPaperBootstrapped: args.onPaperBootstrapped,
		getDocument: () => ({
			blocks: args.blocksRef.current,
			metadata: args.metadata,
		}),
		applyDocument: ({ blocks, metadata }) => {
			args.dispatch({ type: "REPLACE_BLOCKS", blocks });
			args.metadataForm.setFieldValue("title", metadata.title);
			args.metadataForm.setFieldValue("authors", metadata.authors);
			args.metadataForm.setFieldValue("keywords", metadata.keywords);
			args.metadataForm.setFieldValue("abstract", metadata.abstract);
		},
	});

	controller.notifyDocument();

	const syncState = useSelector(controller.store, (state) => state);

	useEffect(
		() => () => {
			controller.dispose();
		},
		[controller],
	);

	return {
		effectivePaperId: controller.effectivePaperId,
		persistEnabled: controller.persistEnabled,
		ready: controller.ready,
		waitingForRemote: controller.waitingForRemote,
		bootstrapError: syncState.bootstrapError,
		saveError: syncState.saveError,
	};
};
