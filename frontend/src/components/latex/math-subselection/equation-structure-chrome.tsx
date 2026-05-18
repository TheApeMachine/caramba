"use client";

import type React from "react";
import {
	type AstNode,
	atomicNode,
	describeAstNode,
	innerPlain,
	type PresetKey,
} from "#/components/latex/math-subselection/ast";
import { STRUCTURE_TRANSFORMS } from "#/components/latex/math-subselection/transforms";
import { Button } from "#/components/ui/button";
import { cn } from "#/lib/utils";

export const STRUCTURE_PRESET_ENTRIES: { key: PresetKey; label: string }[] = [
	{ key: "default", label: "x² + y" },
	{ key: "variance", label: "Variance" },
	{ key: "gauss", label: "Gaussian" },
	{ key: "atom", label: "Just x" },
];

export type EquationStructureChromeProps = {
	disabled?: boolean;
	structureDetached: boolean;
	historyLength: number;
	miniReference: React.RefObject<HTMLSpanElement | null>;
	root: AstNode;
	selectedId: number | null;
	selectedNode: AstNode | null;
	onUndo: () => void;
	onLoadPreset: (key: PresetKey) => void;
	onClearSelection: () => void;
	onApplyTransform: (id: string) => void;
	onRenameAtomic: (value: string) => void;
	onRenameCommit: () => void;
};

/*
EquationStructureChrome is the presentational shell for the math subselection UI.
*/
export function EquationStructureChrome({
	disabled,
	structureDetached,
	historyLength,
	miniReference,
	root,
	selectedId,
	selectedNode,
	onUndo,
	onLoadPreset,
	onClearSelection,
	onApplyTransform,
	onRenameAtomic,
	onRenameCommit,
}: EquationStructureChromeProps) {
	return (
		<div
			className={cn(
				"rounded-xl border border-border/70 bg-muted/25 p-3 shadow-sm",
				disabled && "pointer-events-none opacity-45",
			)}
		>
			<div className="mb-2 flex flex-wrap items-center justify-between gap-2">
				<p className="font-medium text-[11px] text-muted-foreground uppercase tracking-widest">
					Structure
				</p>
				<div className="flex flex-wrap gap-1.5">
					<Button
						className="h-7 px-2 font-mono text-xs"
						disabled={historyLength <= 1 || structureDetached}
						onClick={onUndo}
						type="button"
						variant="ghost"
					>
						Undo
					</Button>
				</div>
			</div>

			{structureDetached ? (
				<div className="rounded-lg border border-dashed border-border bg-background/60 px-3 py-4 text-center text-muted-foreground text-sm">
					<p className="mb-3 text-pretty">
						Source LaTeX no longer matches the built-in structure templates.
						Edit source above, or load a template to use the visual editor
						again.
					</p>
					<Button
						onClick={() => {
							onLoadPreset("default");
						}}
						type="button"
						variant="secondary"
					>
						Reset to x² + y
					</Button>
				</div>
			) : (
				<>
					<div className="mb-3 flex flex-wrap items-center gap-1.5">
						<span className="shrink-0 text-[10px] text-muted-foreground uppercase tracking-wide">
							Start from
						</span>
						{STRUCTURE_PRESET_ENTRIES.map(({ key, label }) => (
							<Button
								className="h-7 rounded-md px-2.5 text-xs"
								key={key}
								onClick={() => {
									onLoadPreset(key);
								}}
								type="button"
								variant="outline"
							>
								{label}
							</Button>
						))}
					</div>

					<div className="mb-3 flex min-h-9 flex-wrap items-center gap-2 rounded-md bg-background/60 px-2.5 py-2 font-mono text-xs">
						{selectedId === null ? (
							<span className="text-muted-foreground">
								Click the equation above to narrow transforms — or apply to the
								whole expression.
							</span>
						) : selectedNode ? (
							<>
								<span className="shrink-0 text-muted-foreground">Selected</span>
								<span ref={miniReference} className="text-foreground" />
								{atomicNode(selectedNode) ? (
									<input
										className="w-20 rounded border border-border bg-background px-2 py-0.5 text-foreground"
										key={selectedNode._id}
										onBlur={onRenameCommit}
										onChange={(event) => {
											onRenameAtomic(event.target.value);
										}}
										defaultValue={
											selectedNode.type === "var"
												? selectedNode.name
												: String(selectedNode.value)
										}
									/>
								) : null}
								<Button
									className="ml-auto h-7 font-mono text-xs"
									onClick={onClearSelection}
									type="button"
									variant="ghost"
								>
									Clear selection
								</Button>
							</>
						) : null}
					</div>

					<p className="mb-1.5 text-[10px] text-muted-foreground uppercase tracking-wide">
						{selectedId === null
							? "Transforms apply to the whole expression"
							: selectedNode
								? `Transforms apply to ${describeAstNode(selectedNode)}`
								: "Transforms apply to the selection"}
					</p>

					<div className="mb-3 flex flex-wrap gap-1.5">
						{STRUCTURE_TRANSFORMS.map((item) => (
							<Button
								className="h-auto gap-1.5 py-1.5 pr-2.5 pl-2"
								key={item.id}
								onClick={() => {
									onApplyTransform(item.id);
								}}
								type="button"
								variant="outline"
							>
								<span className="font-mono text-sm">{item.icon}</span>
								<span className="text-muted-foreground text-xs">
									{item.label}
								</span>
							</Button>
						))}
					</div>

					<div className="flex items-center gap-2 rounded-md bg-muted/40 px-2.5 py-2 font-mono text-[11px]">
						<span className="shrink-0 text-muted-foreground">LaTeX</span>
						<code className="min-w-0 flex-1 break-all text-foreground/90">
							{innerPlain(root)}
						</code>
					</div>
				</>
			)}
		</div>
	);
}
