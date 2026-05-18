import type {
	HeadingLevel,
	PaperBlock,
	PaperHeadingBlock,
	PaperHeadingPresentation,
} from "#/components/latex/model/types";
import { cn } from "#/lib/utils";

const headingScale: Record<HeadingLevel, string> = {
	1: "text-2xl font-semibold tracking-tight sm:text-3xl",
	2: "text-xl font-semibold tracking-tight sm:text-2xl",
	3: "text-lg font-semibold sm:text-xl",
};

/*
followsHeadingPresentation is true when this block sits under the nearest
preceding heading with the given presentation (e.g. abstract body text).
*/
export function followsHeadingPresentation(
	blocks: PaperBlock[],
	blockIndex: number,
	presentation: PaperHeadingPresentation,
): boolean {
	for (let index = blockIndex - 1; index >= 0; index--) {
		const nextBlock = blocks[index];

		if (nextBlock.type === "heading") {
			return nextBlock.presentation === presentation;
		}
	}

	return false;
}

export function headingPresentationLayoutClass(
	block: PaperHeadingBlock,
): string {
	if (block.presentation === "abstract") {
		return cn(
			"min-h-10 border-border/55 border-b py-2.5 text-center font-semibold text-foreground text-sm uppercase tracking-[0.18em] sm:text-base sm:tracking-[0.22em]",
		);
	}

	if (block.presentation === "references") {
		return cn(
			"mt-7 min-h-10 border-border/65 border-t pt-4 font-semibold text-foreground text-xs uppercase tracking-[0.16em] sm:mt-8 sm:text-sm",
		);
	}

	if (block.presentation === "acknowledgments") {
		return cn(
			"min-h-10 py-2 text-center font-medium text-muted-foreground text-sm italic sm:text-[15px]",
		);
	}

	return cn(headingScale[block.level], "min-h-10 py-1.5");
}

export function abstractContinuationLayoutClass(): string {
	return cn(
		"text-[15px] leading-[1.65] text-foreground/95 text-justify sm:px-9 sm:text-[16px] sm:leading-relaxed",
		"px-0.5",
	);
}
