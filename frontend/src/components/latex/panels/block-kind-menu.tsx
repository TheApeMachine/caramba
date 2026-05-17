"use client";

import { Autocomplete as AutocompletePrimitive } from "@base-ui/react/autocomplete";
import {
	ArrowLeftIcon,
	CornerDownLeftIcon,
	SearchIcon,
	SparklesIcon,
} from "lucide-react";
import {
	cloneElement,
	type ReactElement,
	useCallback,
	useEffect,
	useMemo,
	useRef,
	useState,
} from "react";
import { assistantBridge } from "#/components/assistant/assistant-bridge";
import {
	type BlockCatalogGroup,
	type BlockKindDescriptor,
	blockCatalogGroups,
	matchBlockKindQuery,
} from "#/components/latex/model/block-catalog";
import { Button } from "#/components/ui/button";
import {
	Command,
	CommandCollection,
	CommandDialog,
	CommandDialogPopup,
	CommandEmpty,
	CommandFooter,
	CommandGroup,
	CommandGroupLabel,
	CommandInput,
	CommandItem,
	CommandList,
	CommandPanel,
	CommandShortcut,
} from "#/components/ui/command";
import { Input } from "#/components/ui/input";
import { Kbd } from "#/components/ui/kbd";

type BlockKindMenuProps = {
	onSelect: (descriptor: BlockKindDescriptor) => void;
} & (
	| { variant: "trigger"; trigger: ReactElement }
	| {
			variant: "anchored";
			open: boolean;
			onOpenChange: (open: boolean) => void;
	  }
);

type AskAIState = {
	active: boolean;
	query: string;
};

const initialAskAIState: AskAIState = {
	active: false,
	query: "",
};

function FilteredGroups({
	onSelect,
}: {
	onSelect: (descriptor: BlockKindDescriptor) => void;
}) {
	const filteredGroups =
		AutocompletePrimitive.useFilteredItems() as BlockCatalogGroup[];

	const renderItem = (descriptor: BlockKindDescriptor) => (
		<CommandItem
			key={descriptor.id}
			onClick={() => onSelect(descriptor)}
			value={descriptor}
		>
			<span className="flex min-w-0 flex-col">
				<span className="text-foreground text-sm">{descriptor.label}</span>
				<span className="text-muted-foreground text-xs">{descriptor.hint}</span>
			</span>

			{descriptor.shortcut ? (
				<CommandShortcut>{descriptor.shortcut}</CommandShortcut>
			) : null}
		</CommandItem>
	);

	return (
		<>
			{filteredGroups.map((group) => (
				<CommandGroup items={[...group.items]} key={group.label}>
					<CommandGroupLabel>{group.label}</CommandGroupLabel>
					<CommandCollection>{renderItem}</CommandCollection>
				</CommandGroup>
			))}
		</>
	);
}

function BlockKindDialogBody({
	onSelect,
	onClose,
}: {
	onSelect: (descriptor: BlockKindDescriptor) => void;
	onClose: () => void;
}) {
	const [ask, setAsk] = useState<AskAIState>(initialAskAIState);
	const [searchQuery, setSearchQuery] = useState("");
	const searchInputRef = useRef<HTMLInputElement>(null);
	const askInputRef = useRef<HTMLInputElement>(null);
	const commandResetKey = useRef(0);
	const [assistantReady, setAssistantReady] = useState<boolean>(
		() => assistantBridge.get() !== null,
	);

	useEffect(() => {
		const unsubscribe = assistantBridge.subscribe(setAssistantReady);

		return () => {
			if (typeof unsubscribe === "function") {
				unsubscribe();
			}
		};
	}, []);

	const exitAskMode = useCallback(() => {
		setAsk(initialAskAIState);
		setSearchQuery("");
		commandResetKey.current += 1;
		queueMicrotask(() => searchInputRef.current?.focus());
	}, []);

	const enterAskMode = useCallback((seedQuery: string) => {
		setSearchQuery("");
		setAsk({ active: true, query: seedQuery });
		queueMicrotask(() => askInputRef.current?.focus());
	}, []);

	const submitAsk = useCallback(() => {
		const trimmed = ask.query.trim();
		if (!trimmed) return;

		const bridge = assistantBridge.get();
		if (!bridge) return;

		bridge.send(trimmed);
		bridge.setMode("mini");
		onClose();
	}, [ask.query, onClose]);

	const hasResults = useMemo(
		() =>
			!searchQuery.trim() ||
			blockCatalogGroups.some((group) =>
				group.items.some((descriptor) =>
					matchBlockKindQuery(descriptor, searchQuery),
				),
			),
		[searchQuery],
	);

	// When ask.active is true, handler deliberately runs in capture phase so
	// exitAskMode handles Escape first; other capture handlers will not see it.
	useEffect(() => {
		if (!ask.active) return;

		const handler = (event: KeyboardEvent) => {
			if (event.key === "Escape") {
				event.preventDefault();
				event.stopPropagation();
				exitAskMode();
			}
		};

		document.addEventListener("keydown", handler, true);
		return () => document.removeEventListener("keydown", handler, true);
	}, [ask.active, exitAskMode]);

	if (ask.active) {
		return (
			<Command>
				<div className="flex items-center *:first:flex-1">
					<div className="px-2.5 py-1.5">
						<div className="relative w-full">
							<div
								aria-hidden="true"
								className="pointer-events-none absolute inset-y-0 inset-s-px z-10 flex items-center ps-[calc(--spacing(3)-1px)] opacity-80 [&_svg:not([class*='size-'])]:size-4.5 sm:[&_svg:not([class*='size-'])]:size-4 [&_svg]:-mx-0.5"
							>
								<SparklesIcon />
							</div>
							<Input
								aria-label="Ask the assistant"
								className="border-transparent! bg-transparent! shadow-none before:hidden has-focus-visible:ring-0 *:data-[slot=input]:ps-[calc(--spacing(8.5)-1px)] sm:*:data-[slot=input]:ps-[calc(--spacing(8)-1px)]"
								onChange={(event) =>
									setAsk((prev) => ({ ...prev, query: event.target.value }))
								}
								onKeyDown={(event) => {
									if (event.key === "Enter") {
										event.preventDefault();
										submitAsk();
									}
								}}
								placeholder="Ask the assistant to write or edit…"
								ref={askInputRef}
								size="lg"
								value={ask.query}
							/>
						</div>
					</div>
					<Button
						className="me-2.5 rounded-md text-sm not-hover:text-muted-foreground sm:text-xs"
						onClick={exitAskMode}
						size="sm"
						variant="ghost"
					>
						<ArrowLeftIcon className="size-4 sm:size-3.5" />
						Back
						<Kbd className="-me-1.5 ms-0.5">Esc</Kbd>
					</Button>
				</div>

				<CommandPanel className="p-5">
					<p className="text-muted-foreground text-sm">
						The assistant will use the current paper context to perform the
						task. Press <Kbd>Enter</Kbd> to send, then watch the assistant
						panel.
					</p>

					{!assistantReady ? (
						<p className="mt-3 text-destructive text-xs">
							Assistant is not available. Open the assistant panel first, then
							retry.
						</p>
					) : null}
				</CommandPanel>

				<CommandFooter>
					<div className="flex items-center gap-2">
						<Kbd>
							<CornerDownLeftIcon />
						</Kbd>
						<span>Send to assistant</span>
					</div>
				</CommandFooter>
			</Command>
		);
	}

	return (
		// Command remounts via key={commandResetKey.current}; exitAskMode increments
		// the ref to reset Command state without storing a render-only counter.
		<Command
			filter={(item, query) =>
				matchBlockKindQuery(item as BlockKindDescriptor, query)
			}
			items={blockCatalogGroups}
			key={commandResetKey.current}
		>
			<div className="relative flex items-center *:first:flex-1">
				<CommandInput
					onChange={(event) => setSearchQuery(event.target.value)}
					onKeyDown={(event) => {
						if (event.key === "Tab") {
							event.preventDefault();
							enterAskMode(searchQuery);
							return;
						}

						if (event.key === "Enter" && !hasResults && searchQuery.trim()) {
							event.preventDefault();
							enterAskMode(searchQuery);
						}
					}}
					placeholder="Filter blocks or ask the assistant…"
					ref={searchInputRef}
					value={searchQuery}
				/>
				<Button
					className="me-2.5 rounded-md text-sm not-hover:text-muted-foreground sm:text-xs"
					disabled={!assistantReady}
					onClick={() => enterAskMode(searchQuery)}
					size="sm"
					variant="ghost"
				>
					<SparklesIcon className="size-4 sm:size-3.5" />
					Ask AI
					<Kbd className="-me-1.5 ms-0.5">Tab</Kbd>
				</Button>
			</div>

			<CommandPanel>
				<CommandEmpty className="not-empty:py-12">
					{searchQuery.trim() ? (
						<div className="flex flex-col flex-wrap items-center gap-2 wrap-break-word">
							<SearchIcon className="size-5 text-muted-foreground" />
							<p>No matching blocks.</p>
							<p>
								Press <Kbd>Enter</Kbd> to ask the assistant about:
								<br />{" "}
								<strong className="font-medium text-foreground">
									{searchQuery}
								</strong>
							</p>
						</div>
					) : null}
				</CommandEmpty>

				<CommandList>
					<FilteredGroups
						onSelect={(descriptor) => {
							onSelect(descriptor);
							onClose();
						}}
					/>
				</CommandList>
			</CommandPanel>

			<CommandFooter>
				<div className="flex items-center gap-2">
					<Kbd>
						<CornerDownLeftIcon />
					</Kbd>
					<span>Insert</span>
				</div>
				<div className="flex items-center gap-2">
					<Kbd>Esc</Kbd>
					<span>Close</span>
				</div>
			</CommandFooter>
		</Command>
	);
}

function TriggeredBlockKindMenu({
	trigger,
	onSelect,
}: {
	trigger: ReactElement;
	onSelect: (descriptor: BlockKindDescriptor) => void;
}) {
	const [open, setOpen] = useState(false);

	const triggerWithHandler = cloneElement(
		trigger as ReactElement<{ onClick?: () => void }>,
		{ onClick: () => setOpen(true) },
	);

	return (
		<>
			{triggerWithHandler}
			<CommandDialog onOpenChange={setOpen} open={open}>
				<CommandDialogPopup>
					<BlockKindDialogBody
						onClose={() => setOpen(false)}
						onSelect={onSelect}
					/>
				</CommandDialogPopup>
			</CommandDialog>
		</>
	);
}

function AnchoredBlockKindMenu({
	open,
	onOpenChange,
	onSelect,
}: {
	open: boolean;
	onOpenChange: (open: boolean) => void;
	onSelect: (descriptor: BlockKindDescriptor) => void;
}) {
	return (
		<CommandDialog onOpenChange={onOpenChange} open={open}>
			<CommandDialogPopup>
				<BlockKindDialogBody
					onClose={() => onOpenChange(false)}
					onSelect={onSelect}
				/>
			</CommandDialogPopup>
		</CommandDialog>
	);
}

export function BlockKindMenu(props: BlockKindMenuProps) {
	if (props.variant === "trigger") {
		return (
			<TriggeredBlockKindMenu
				onSelect={props.onSelect}
				trigger={props.trigger}
			/>
		);
	}

	return (
		<AnchoredBlockKindMenu
			onOpenChange={props.onOpenChange}
			onSelect={props.onSelect}
			open={props.open}
		/>
	);
}
