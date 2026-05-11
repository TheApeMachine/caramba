import { Plus, Trash2 } from "lucide-react";
import { Button } from "#/components/ui/button";
import { Input } from "#/components/ui/input";
import { Label } from "#/components/ui/label";
import {
	Select,
	SelectItem,
	SelectPopup,
	SelectTrigger,
	SelectValue,
} from "#/components/ui/select";
import { Slider } from "#/components/ui/slider";
import { Textarea } from "#/components/ui/textarea";
import type { Persona, Provider, Session } from "../types";
import { AVAILABLE_MODELS, DEFAULT_PERSONA } from "../types";

const PROVIDER_LABELS: Record<Provider, string> = {
	openai: "OpenAI",
	anthropic: "Anthropic",
	google: "Google",
	xai: "xAI",
};

const PROVIDERS = Array.from(
	new Set(AVAILABLE_MODELS.map((m) => m.provider)),
) as Provider[];

function sliderScalar(next: number | readonly number[]): number | undefined {
	if (typeof next === "number") {
		return next;
	}

	return next[0];
}

type Props = {
	session: Session;
	onUpdatePersona: (persona: Persona) => void;
	onAddPersona: (persona: Persona) => void;
	onRemovePersona: (id: string) => void;
	onWindowSizeChange: (size: number) => void;
};

function PersonaCard({
	persona,
	removable,
	onUpdate,
	onRemove,
}: {
	persona: Persona;
	removable: boolean;
	onUpdate: (p: Persona) => void;
	onRemove: () => void;
}) {
	return (
		<div className="rounded-xl border bg-muted/30 p-4 flex flex-col gap-3">
			<div className="flex items-center gap-2">
				<Input
					value={persona.name}
					onChange={(e) => onUpdate({ ...persona, name: e.target.value })}
					className="h-7 text-sm font-medium flex-1"
					placeholder="Persona name"
				/>
				{removable && (
					<Button size="icon-xs" variant="ghost" onClick={onRemove}>
						<Trash2 />
					</Button>
				)}
			</div>

			<Textarea
				value={persona.systemPrompt}
				onChange={(e) => onUpdate({ ...persona, systemPrompt: e.target.value })}
				placeholder="System prompt…"
				className="text-xs min-h-[72px]"
			/>

			<div className="flex flex-col gap-2">
				<Label className="text-xs text-muted-foreground">Model</Label>
				<Select
					value={persona.model}
					onValueChange={(v) => v && onUpdate({ ...persona, model: v })}
				>
					<SelectTrigger size="sm" className="w-full">
						<SelectValue />
					</SelectTrigger>
					<SelectPopup>
						{PROVIDERS.map((provider) => (
							<div key={provider}>
								<div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
									{PROVIDER_LABELS[provider]}
								</div>
								{AVAILABLE_MODELS.filter((m) => m.provider === provider).map(
									(m) => (
										<SelectItem key={m.id} value={m.id}>
											{m.label}
										</SelectItem>
									),
								)}
							</div>
						))}
					</SelectPopup>
				</Select>
			</div>

			<div className="flex flex-col gap-1">
				<div className="flex items-center justify-between">
					<Label className="text-xs text-muted-foreground">Temperature</Label>
					<span className="text-xs tabular-nums">
						{persona.temperature.toFixed(1)}
					</span>
				</div>
				<Slider
					min={0}
					max={2}
					step={0.1}
					value={[persona.temperature]}
					onValueChange={(next) => {
						const v = sliderScalar(next);
						onUpdate({ ...persona, temperature: v ?? persona.temperature });
					}}
				/>
			</div>

			<div className="flex flex-col gap-1">
				<div className="flex items-center justify-between">
					<Label className="text-xs text-muted-foreground">Max tokens</Label>
					<span className="text-xs tabular-nums">{persona.maxTokens}</span>
				</div>
				<Slider
					min={256}
					max={8192}
					step={256}
					value={[persona.maxTokens]}
					onValueChange={(next) => {
						const v = sliderScalar(next);
						onUpdate({ ...persona, maxTokens: v ?? persona.maxTokens });
					}}
				/>
			</div>
		</div>
	);
}

export function SettingsPanel({
	session,
	onUpdatePersona,
	onAddPersona,
	onRemovePersona,
	onWindowSizeChange,
}: Props) {
	const handleAdd = () => {
		onAddPersona({
			...DEFAULT_PERSONA,
			id: crypto.randomUUID(),
			name: `Researcher ${session.personas.length + 1}`,
			systemPrompt:
				"You are a specialist researcher. Build on the conversation and add new insights.",
		});
	};

	return (
		<div className="flex flex-col gap-6 p-4 overflow-y-auto">
			<div className="flex flex-col gap-3">
				<div className="flex items-center justify-between">
					<span className="text-sm font-medium">Personas</span>
					<Button size="xs" variant="outline" onClick={handleAdd}>
						<Plus />
						Add persona
					</Button>
				</div>

				{session.personas.map((persona) => (
					<PersonaCard
						key={persona.id}
						persona={persona}
						removable={session.personas.length > 1}
						onUpdate={onUpdatePersona}
						onRemove={() => onRemovePersona(persona.id)}
					/>
				))}
			</div>

			<div className="flex flex-col gap-2 border-t pt-4">
				<div className="flex items-center justify-between">
					<Label className="text-xs text-muted-foreground">
						Context window (messages)
					</Label>
					<span className="text-xs tabular-nums">{session.windowSize}</span>
				</div>
				<Slider
					min={4}
					max={100}
					step={4}
					value={[session.windowSize]}
					onValueChange={(next) => {
						const v = sliderScalar(next);
						if (v !== undefined) {
							onWindowSizeChange(v);
						}
					}}
				/>
				<p className="text-xs text-muted-foreground">
					First message is always pinned. Remaining context slides over the last
					N messages.
				</p>
			</div>
		</div>
	);
}
