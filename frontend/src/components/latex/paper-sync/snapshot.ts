import { serializePaperDocument } from "#/components/latex/model/paper-document";
import type { PaperMetadata } from "#/components/latex/model/types";

/*
paperMetadataSnapshot is the canonical string the controller uses to
detect metadata changes between renders. Equal strings mean "nothing
worth saving"; the autosave timer only restarts when this differs from
the previous snapshot. Blocks live in their own collection now and are
not part of this snapshot.
*/
export const paperMetadataSnapshot = (metadata: PaperMetadata): string =>
	JSON.stringify(serializePaperDocument(metadata));
