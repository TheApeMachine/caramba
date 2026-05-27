import type { MessagePart } from "@tanstack/ai-client";
import { Brain, ChevronDown, Sparkles, Wrench } from "lucide-react";
import { useTranslation } from "react-i18next";
import ReactMarkdown from "react-markdown";
import { Avatar, AvatarFallback } from "#/components/ui/avatar";
import {
	Collapsible,
	CollapsiblePanel,
	CollapsibleTrigger,
} from "#/components/ui/collapsible";
import { ScrollArea } from "#/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { UIMessage } from "../types";

function partKey(messageId: string, part: MessagePart, index: number): string {
	if (part.type === "tool-call") return `${messageId}:tool-call:${part.id}`;
	if (part.type === "tool-result")
		return `${messageId}:tool-result:${part.toolCallId}`;
	return `${messageId}:${part.type}:${index}`;
}

function ThinkingPart({ content }: { content: string }) {
	const { t } = useTranslation();

	return (
		<Collapsible className="mb-2">
			<CollapsibleTrigger className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer group">
				<Brain className="size-3 shrink-0 text-brand" />
				<span className="italic">{t("assistant.reasoning")}</span>
				<ChevronDown className="size-3 transition-transform group-data-open:rotate-180" />
			</CollapsibleTrigger>
			<CollapsiblePanel>
				<div className="mt-1.5 text-xs text-muted-foreground leading-relaxed border-l-2 border-brand/40 pl-2.5 prose prose-xs dark:prose-invert max-w-none prose-p:my-1">
					<ReactMarkdown>{content}</ReactMarkdown>
				</div>
			</CollapsiblePanel>
		</Collapsible>
	);
}

function ToolCallPart({ name, args }: { name: string; args: string }) {
	let formatted = args;
	try {
		formatted = JSON.stringify(JSON.parse(args), null, 2);
	} catch {
		/* leave as-is */
	}

	return (
		<Collapsible className="mb-1.5">
			<CollapsibleTrigger className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer group w-full">
				<Wrench className="size-3 shrink-0 text-info" />
				<span className="font-medium text-info-foreground truncate">
					{name}
				</span>
				<ChevronDown className="size-3 ml-auto shrink-0 transition-transform group-data-open:rotate-180" />
			</CollapsibleTrigger>
			<CollapsiblePanel>
				<pre className="mt-1.5 text-[11px] text-info-foreground whitespace-pre-wrap break-all border-l-2 border-info/40 pl-2.5 leading-relaxed">
					{formatted}
				</pre>
			</CollapsiblePanel>
		</Collapsible>
	);
}

function ToolResultPart({
	content,
	error,
}: {
	content: unknown;
	error?: string | null;
}) {
	const { t } = useTranslation();
	const raw =
		error ??
		(typeof content === "string" ? content : JSON.stringify(content, null, 2));
	let formatted = raw;
	try {
		formatted = JSON.stringify(JSON.parse(raw), null, 2);
	} catch {
		/* leave as-is */
	}

	const isError = Boolean(error);

	return (
		<Collapsible className="mb-1.5" defaultOpen={isError}>
			<CollapsibleTrigger className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer group w-full">
				<span
					className={cn(
						"size-2 rounded-full shrink-0",
						isError ? "bg-destructive" : "bg-success",
					)}
				/>
				<span
					className={cn(
						"font-medium truncate",
						isError ? "text-destructive" : "text-success-foreground",
					)}
				>
					{isError ? t("assistant.toolError") : t("assistant.toolResult")}
				</span>
				<ChevronDown className="size-3 ml-auto shrink-0 transition-transform group-data-open:rotate-180" />
			</CollapsibleTrigger>
			<CollapsiblePanel>
				<pre
					className={cn(
						"mt-1.5 text-[11px] whitespace-pre-wrap break-all border-l-2 pl-2.5 leading-relaxed",
						isError
							? "text-destructive border-destructive/40"
							: "text-success-foreground border-success/40",
					)}
				>
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
						<div
							key={key}
							className="text-sm leading-relaxed prose prose-sm dark:prose-invert max-w-none prose-p:my-2 prose-pre:my-2 prose-ul:my-2 prose-ol:my-2 prose-headings:my-2"
						>
							<ReactMarkdown>{part.content}</ReactMarkdown>
						</div>
					);
				}
				if (part.type === "tool-call") {
					return (
						<ToolCallPart key={key} name={part.name} args={part.arguments} />
					);
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
	reasoningActive?: boolean;
	isSubmitted?: boolean;
	compact?: boolean;
};

export function MessageFeed({
	messages,
	reasoningActive,
	isSubmitted,
	compact,
}: Props) {
	const { t } = useTranslation();

	return (
		<ScrollArea aria-label="Conversation">
			<section className="flex flex-col-reverse">
				{isSubmitted && (
					<div className="mb-4 mr-auto">
						<div className="bg-muted rounded-xl px-3 py-2 flex items-center gap-2">
							{reasoningActive ? (
								<>
									<Sparkles className="size-3.5 text-brand animate-pulse" />
									<span className="text-xs italic bg-linear-to-r from-muted-foreground/40 via-foreground to-muted-foreground/40 bg-size-[200%_100%] bg-clip-text text-transparent animate-[shimmer_2s_linear_infinite]">
										{t("assistant.thinking")}
									</span>
								</>
							) : (
								<span className="flex gap-1">
									<span className="size-1.5 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:0ms]" />
									<span className="size-1.5 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:150ms]" />
									<span className="size-1.5 rounded-full bg-muted-foreground/60 animate-bounce [animation-delay:300ms]" />
								</span>
							)}
						</div>
					</div>
				)}

				{[...messages].reverse().map((message) => {
					const isUser = message.role === "user";
					const label = isUser
						? t("assistant.you")
						: (message.personaName ?? t("assistant.title"));
					const initials = label
						.split(/\s+/)
						.map((w) => w[0])
						.slice(0, 2)
						.join("")
						.toUpperCase();
					return (
						<div
							key={message.id}
							className={cn(
								"mb-4 flex gap-2 items-start",
								isUser ? "flex-row-reverse" : "flex-row",
							)}
						>
							<Avatar
								className={cn(
									"size-7 shrink-0 mt-0.5",
									isUser
										? "bg-primary text-primary-foreground"
										: "bg-brand/15 text-brand",
								)}
							>
								<AvatarFallback
									className={cn(
										isUser
											? "bg-primary text-primary-foreground"
											: "bg-brand/15 text-brand",
									)}
								>
									{initials}
								</AvatarFallback>
							</Avatar>
							<div
								className={cn(
									"flex flex-col min-w-0",
									isUser
										? "items-end max-w-[80%]"
										: "items-start max-w-[85%] w-full",
								)}
							>
								<span className="text-[11px] text-muted-foreground mb-0.5 px-1">
									{label}
								</span>
								<div
									className={cn(
										"rounded-xl px-3 py-2 text-sm",
										isUser
											? "bg-primary text-primary-foreground rounded-tr-sm"
											: "bg-muted rounded-tl-sm",
									)}
								>
									<MessageParts message={message} />
								</div>
							</div>
						</div>
					);
				})}

				{messages.length === 0 && (
					<p className="text-xs text-muted-foreground text-center mt-8">
						{compact ? t("assistant.readySingle") : t("assistant.readyTeam")}
					</p>
				)}
			</section>
		</ScrollArea>
	);
}
