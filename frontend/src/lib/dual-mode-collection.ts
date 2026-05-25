import { electricCollectionOptions } from "@tanstack/electric-db-collection";
import {
	createCollection,
	localStorageCollectionOptions,
} from "@tanstack/react-db";
import type { z } from "zod";
import {
	electricAwaitOptions,
	type SyncMode,
	shapeUrl,
} from "./electric-shape";

type DualModeCollectionCache = {
	cloud?: ReturnType<typeof createCollection>;
	local?: ReturnType<typeof createCollection>;
};

const collectionCaches = new Map<string, DualModeCollectionCache>();

export type DualModeCollectionConfig<Row extends Record<string, unknown>> = {
	cacheKey: string;
	schema: z.ZodType<Row>;
	getKey: (item: Row) => string;
	cloud: {
		id: string;
		shapePath: string;
		parser?: { timestamptz: (value: string) => Date };
		onInsert?: Parameters<typeof electricCollectionOptions<Row>>[0]["onInsert"];
		onUpdate?: Parameters<typeof electricCollectionOptions<Row>>[0]["onUpdate"];
		onDelete?: Parameters<typeof electricCollectionOptions<Row>>[0]["onDelete"];
	};
	local: {
		id: string;
		storageKey: string;
	};
};

/*
createDualModeCollection centralizes local vs Electric routing so each entity
defines its schema once and registers cloud mutation handlers in one place.
*/
export function createDualModeCollection<Row extends Record<string, unknown>>(
	config: DualModeCollectionConfig<Row>,
) {
	return (mode: SyncMode) => {
		let cache = collectionCaches.get(config.cacheKey);

		if (!cache) {
			cache = {};
			collectionCaches.set(config.cacheKey, cache);
		}

		if (mode === "local") {
			cache.local ??= createCollection(
				localStorageCollectionOptions({
					id: config.local.id,
					storageKey: config.local.storageKey,
					schema: config.schema,
					getKey: config.getKey,
				}),
			);

			return cache.local;
		}

		cache.cloud ??= createCollection(
			electricCollectionOptions({
				id: config.cloud.id,
				schema: config.schema,
				getKey: config.getKey,
				shapeOptions: {
					url: shapeUrl(config.cloud.shapePath),
					parser: config.cloud.parser,
				},
				onInsert: config.cloud.onInsert,
				onUpdate: config.cloud.onUpdate,
				onDelete: config.cloud.onDelete,
			}),
		);

		return cache.cloud;
	};
}

export { electricAwaitOptions };
