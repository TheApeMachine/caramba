import type { MessagePart, UIMessage } from "@tanstack/ai-client";
import { Brain, ChevronDown, Wrench } from "lucide-react";
import {
	Collapsible,
	CollapsiblePanel,
	CollapsibleTrigger,
} from "#/components/ui/collapsible";
import { cn } from "@/lib/utils";

function partKey(messageId: string, part: MessagePart, index: number): string {
	if (part.type === "tool-call") return `${messageId}:tool-call:${part.id}`;
	if (part.type === "tool-result") return `${messageId}:tool-result:${part.toolCallId}`;
	return `${messageId}:${part.type}:${index}`;
}

function ThinkingPart({ content }: { content: string }) {
	return (
		<Collapsible className="mb-2">
			<CollapsibleTrigger className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer group">
				<Brain className="size-3 shrink-0 text-violet-400" />
				<span className="italic">Reasoning</span>
				<ChevronDown className="size-3 transition-transform group-data-open:rotate-180" />
			</CollapsibleTrigger>
			<CollapsiblePanel>
				<pre className="mt-1.5 text-xs text-muted-foreground whitespace-pre-wrap leading-relaxed border-l-2 border-violet-300/40 pl-2.5">
					{content}
				</pre>
			</CollapsiblePanel>
		</Collapsible>
	);
}

function ToolCallPart({ name, args }: { name: string; args: string }) {
	let formatted = args;
	try {
		formatted = JSON.stringify(JSON.parse(args), null, 2);
	} catch { /* leave as-is */ }

	return (
		<Collapsible className="mb-1.5">
			<CollapsibleTrigger className="flex items-center gap-1.5 text-xs hover:text-foreground transition-colors cursor-pointer group w-full">
				<Wrench className="size-3 shrink-0 text-blue-400" />
				<span className="font-medium text-blue-600 dark:text-blue-400 truncate">{name}</span>
				<ChevronDown className="size-3 ml-auto shrink-0 transition-transform group-data-open:rotate-180" />
			</CollapsibleTrigger>
			<CollapsiblePanel>
				<pre className="mt-1.5 text-[11px] text-blue-700 dark:text-blue-300 whitespace-pre-wrap break-all border-l-2 border-blue-300/40 pl-2.5 leading-relaxed">
					{formatted}
				</pre>
			</CollapsiblePanel>
		</Collapsible>
	);
}

function ToolResultPart({ content, error }: { content: unknown; error?: string | null }) {
	const raw = error ?? (typeof content === "string" ? content : JSON.stringify(content, null, 2));
	let formatted = raw;
	try {
		formatted = JSON.stringify(JSON.parse(raw), null, 2);
	} catch { /* leave as-is */ }

	const isError = Boolean(error);

	return (
		<Collapsible className="mb-1.5" defaultOpen={isError}>
			<CollapsibleTrigger className="flex items-center gap-1.5 text-xs hover:text-foreground transition-colors cursor-pointer group w-full">
				<span className={cn("size-2 rounded-full shrink-0", isError ? "bg-destructive" : "bg-emerald-400")} />
				<span className={cn("font-medium truncate", isError ? "text-destructive" : "text-emerald-600 dark:text-emerald-400")}>
					{isError ? "Tool error" : "Tool result"}
				</span>
				<ChevronDown className="size-3 ml-auto shrink-0 transition-transform group-data-open:rotate-180" />
			</CollapsibleTrigger>
			<CollapsiblePanel>
				<pre className={cn(
					"mt-1.5 text-[11px] whitespace-pre-wrap break-all border-l-2 pl-2.5 leading-relaxed",
					isError
						? "text-destructive border-destructive/40"
						: "text-emerald-700 dark:text-emerald-300 border-emerald-300/40",
				)}>
					{formatted}
				</pre>
			</CollapsiblePanel>
		</Collapsible>
	);
}

function MessageParts({ message }: { message: UIMessage }) {
	return (
		<div className="flex flex-col">
			{message.parts.map((part, index) => {
				const key = partKey(message.id, part, index);

				if (part.type === "thinking") {
					return <ThinkingPart key={key} content={part.content} />;
				}
				if (part.type === "text") {
					return (
						<p key={key} className="text-sm leading-relaxed whitespace-pre-wrap">
							{part.content}
						</p>
					);
				}
				if (part.type === "tool-call") {
					return <ToolCallPart key={key} name={part.name} args={part.arguments} />;
				}
				if (part.type === "tool-result") {
					return (
						<ToolResultPart
							key={key}
							content={part.content}
							error={part.error}
						/>
					);
				}
				return null;
			})}
		</div>
	);
}

type Props = {
	messages: UIMessage[];
	streamingPersonaId?: string | null;
	isSubmitted?: boolean;
	compact?: boolean;
};

export function MessageFeed({ messages, isSubmitted, compact }: Props) {
	return (
		<section
			className={cn(
				"flex flex-col-reverse overflow-y-auto min-h-0",
				compact ? "flex-1 px-4 py-3" : "flex-1 px-6 py-4",
			)}
			aria-label="Conversation"
		>
			{isSubmitted && (
				<div className="mb-4 mr-auto">
					<div className="bg-muted rounded-xl px-3 py-2">
						<span className="flex gap-1">
							<span className="size-1.5 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:0ms]" />
							<span className="size-1.5 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:150ms]" />
							<span className="size-1.5 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:300ms]" />
						</span>
					</div>
				</div>
			)}

			{[...messages].reverse().map((message) => (
				<div
					key={message.id}
					className={cn(
						"mb-4",
						message.role === "user"
							? "ml-auto max-w-[80%] text-right"
							: "mr-auto max-w-[88%] w-full",
					)}
				>
					<div
						className={cn(
							"rounded-xl px-3 py-2 text-sm",
							message.role === "user"
								? "inline-block bg-primary text-primary-foreground"
								: "bg-muted",
						)}
					>
						<MessageParts message={message} />
					</div>
				</div>
			))}

			{messages.length === 0 && (
				<p className="text-xs text-muted-foreground text-center mt-8">
					{compact
						? "Ask anything — I'm ready."
						: "Ask anything — your research team is ready."}
				</p>
			)}
		</section>
	);
}
