"use client";

import { FileCodeIcon } from "lucide-react";
import { specToManifest } from "./manifest";
import type { BenchmarkSpec } from "./model";

interface ManifestPreviewProps {
	spec: BenchmarkSpec;
}

/*
ManifestPreview renders the wizard's working YAML side-by-side with the
form. Researchers think in manifests; seeing the artifact build itself as
they click removes the "what is this actually going to run?" gap.
*/
export const ManifestPreview = ({ spec }: ManifestPreviewProps) => {
	const yaml = specToManifest(spec);

	return (
		<div className="flex h-full flex-col overflow-hidden rounded-2xl border bg-muted/30">
			<header className="flex items-center justify-between gap-2 border-b bg-muted/40 px-4 py-2.5">
				<div className="flex items-center gap-2 text-sm">
					<FileCodeIcon className="size-4 text-muted-foreground" />
					<span className="font-medium">manifest.yml</span>
				</div>
				<span className="font-mono text-muted-foreground text-xs">
					{yaml.split("\n").length} lines
				</span>
			</header>
			<pre className="flex-1 overflow-auto px-4 py-3 font-mono text-foreground/85 text-xs leading-relaxed">
				{yaml}
			</pre>
		</div>
	);
};
