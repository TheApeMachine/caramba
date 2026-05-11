import type { MessagePart, UIMessage } from "@tanstack/ai-client";
import { cn } from "@/lib/utils";

function partKey(messageId: string, part: MessagePart, index: number): string {
	if (part.type === "tool-call") return `${messageId}:tool-call:${part.id}`;
	if (part.type === "tool-result") return `${messageId}:tool-result:${part.toolCallId}`;
	return `${messageId}:${part.type}:${index}`;
}

function MessageParts({ message }: { message: UIMessage }) {
	return (
		<>
			{message.parts.map((part, index) => {
				const key = partKey(message.id, part, index);

				if (part.type === "thinking") {
					return (
						<p key={key} className="text-xs text-muted-foreground italic mb-1">
							{part.content}
						</p>
					);
				}
				if (part.type === "text") {
					return (
						<p key={key} className="text-sm leading-relaxed whitespace-pre-wrap">
							{part.content}
						</p>
					);
				}
				if (part.type === "tool-call") {
					return (
						<div key={key} className="rounded border border-blue-200 bg-blue-50 dark:border-blue-900 dark:bg-blue-950/40 p-2 mb-1 text-xs">
							<p className="font-semibold text-blue-800 dark:text-blue-300 mb-0.5">⚙ {part.name}</p>
							<pre className="whitespace-pre-wrap break-all text-blue-700 dark:text-blue-400">{part.arguments}</pre>
						</div>
					);
				}
				if (part.type === "tool-result") {
					return (
						<div key={key} className="rounded border border-emerald-200 bg-emerald-50 dark:border-emerald-900 dark:bg-emerald-950/40 p-2 mb-1 text-xs">
							<p className="font-semibold text-emerald-800 dark:text-emerald-300 mb-0.5">✓ Result</p>
							<pre className="whitespace-pre-wrap break-all text-emerald-700 dark:text-emerald-400">
								{part.error ?? (typeof part.content === "string" ? part.content : JSON.stringify(part.content, null, 2))}
							</pre>
						</div>
					);
				}
				return null;
			})}
		</>
	);
}

type Props = {
	messages: UIMessage[];
	streamingPersonaId?: string | null;
	isSubmitted?: boolean;
	compact?: boolean;
};

export function MessageFeed({ messages, streamingPersonaId, isSubmitted, compact }: Props) {
	return (
		<section
			className={cn(
				"flex flex-col-reverse overflow-y-auto min-h-0",
				compact ? "flex-1 px-4 py-3" : "flex-1 px-6 py-4",
			)}
			aria-label="Conversation"
		>
			{/* Typing indicator — only when waiting for first chunk */}
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
							: "mr-auto max-w-[88%]",
					)}
				>
					<div
						className={cn(
							"inline-block rounded-xl px-3 py-2 text-sm",
							message.role === "user"
								? "bg-primary text-primary-foreground"
								: "bg-muted",
						)}
					>
						<MessageParts message={message} />
					</div>
				</div>
			))}

			{messages.length === 0 && (
				<p className="text-xs text-muted-foreground text-center mt-8">
					Ask anything — your research team is ready.
				</p>
			)}
		</section>
	);
}
