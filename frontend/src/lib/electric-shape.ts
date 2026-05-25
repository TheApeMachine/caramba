export type SyncMode = "cloud" | "local";

export const shapeUrl = (path: string): string => {
	if (typeof window === "undefined") {
		return `http://localhost/api/shape/${path}`;
	}

	return `${window.location.origin}/api/shape/${path}`;
};

export const electricAwaitOptions = (txid: number | undefined) => {
	const skipTxidAwait =
		import.meta.env.VITE_ELECTRIC_SKIP_TXID_AWAIT === "true";

	if (skipTxidAwait || typeof txid !== "number") {
		return undefined;
	}

	return { timeout: 60_000, txid };
};
