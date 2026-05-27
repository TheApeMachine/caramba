import { serializePaperDocument } from "#/components/latex/model/paper-document";
import type { PaperBlock, PaperMetadata } from "#/components/latex/model/types";

/*
paperDocumentSnapshot is the canonical string the controller uses to
detect document changes between renders. Equal strings mean "nothing
worth saving"; the autosave timer only restarts when this differs
from the previous snapshot.
*/
export const paperDocumentSnapshot = (
	metadata: PaperMetadata,
	blocks: PaperBlock[],
): string =>
	JSON.stringify(serializePaperDocument(metadata, blocks));

/*
paperStructureSignature is the join of block ids in order. When this
changes between snapshots, the editor performed a structural edit
(insert / remove / reorder) and the controller shortens the autosave
debounce so the layout reflects the persisted state quickly.
*/
export const paperStructureSignature = (blocks: PaperBlock[]): string =>
	blocks.map((block) => block.id).join("|");
