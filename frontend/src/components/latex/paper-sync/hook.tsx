"use client";

import { useSelector } from "@tanstack/react-store";
import { useEffect, useRef } from "react";
import {
	createPaperSyncController,
	type PaperSyncController,
} from "#/components/latex/paper-sync/controller";
import type { PaperMetadata } from "#/components/latex/model/types";

type MetadataWriter = {
	setFieldValue: (name: keyof PaperMetadata, value: string) => void;
};

export type UseResearchPaperSyncArgs = {
	paperIdProp?: string;
	bootstrapProjectId?: string;
	onPaperBootstrapped?: (paperId: string) => void;
	onBootstrapPaperCreated?: (paperId: string) => void;
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
useResearchPaperSync is a thin React wrapper over PaperSyncController.
The controller owns every piece of sync state (Tanstack Store +
collection subscription), so this hook only threads props through,
subscribes to the controller's state, and disposes the controller on
unmount.
*/
export const useResearchPaperSync = (
	args: UseResearchPaperSyncArgs,
): ResearchPaperSyncResult => {
	const controllerRef = useRef<PaperSyncController | null>(null);

	if (controllerRef.current === null) {
		controllerRef.current = createPaperSyncController({
			paperIdProp: args.paperIdProp,
			bootstrapProjectId: args.bootstrapProjectId,
			onPaperBootstrapped: args.onPaperBootstrapped,
			onBootstrapPaperCreated: args.onBootstrapPaperCreated,
			getMetadata: () => args.metadata,
			applyMetadata: (metadata) => {
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
		onBootstrapPaperCreated: args.onBootstrapPaperCreated,
		getMetadata: () => args.metadata,
		applyMetadata: (metadata) => {
			args.metadataForm.setFieldValue("title", metadata.title);
			args.metadataForm.setFieldValue("authors", metadata.authors);
			args.metadataForm.setFieldValue("keywords", metadata.keywords);
			args.metadataForm.setFieldValue("abstract", metadata.abstract);
		},
	});

	controller.notifyMetadata();

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
