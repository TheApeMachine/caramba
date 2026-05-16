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
import { Switch } from "#/components/ui/switch";
import { Textarea } from "#/components/ui/textarea";
import { useAssistantMode } from "../use-assistant-mode";
import {
	type AdapterType,
	AVAILABLE_MODELS,
	DEFAULT_PERSONA,
	type Persona,
	type PersonaScope,
	type Provider,
	type Session,
} from "../types";

const PROVIDER_LABELS: Record<Provider, string> = {
	openai: "OpenAI",
	anthropic: "Anthropic",
	google: "Google",
	xai: "xAI",
};

const PROVIDERS = Array.from(
	new Set(AVAILABLE_MODELS.map((m) => m.provider)),
) as Provider[];

const SCOPE_LABELS: Record<PersonaScope, string> = {
	global: "Global",
	team: "Team",
	personal: "Personal",
};

const ADAPTER_LABELS: Record<AdapterType, string> = {
	openai: "OpenAI (cloud)",
	ollama: "Ollama (local)",
	"openai-compat": "OpenAI-compatible",
};

function sliderScalar(next: number | readonly number[]): number | undefined {
	if (typeof next === "number") return next;
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
	const showEndpoint =
		persona.adapterType === "ollama" || persona.adapterType === "openai-compat";

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

			<div className="grid grid-cols-2 gap-2">
				<div className="flex flex-col gap-1">
					<Label className="text-xs text-muted-foreground">Scope</Label>
					<Select
						value={persona.scope}
						onValueChange={(v) =>
							v && onUpdate({ ...persona, scope: v as PersonaScope })
						}
					>
						<SelectTrigger size="sm" className="w-full">
							<SelectValue />
						</SelectTrigger>
						<SelectPopup>
							<SelectItem value="personal">{SCOPE_LABELS.personal}</SelectItem>
							<SelectItem value="team">{SCOPE_LABELS.team}</SelectItem>
							<SelectItem value="global">{SCOPE_LABELS.global}</SelectItem>
						</SelectPopup>
					</Select>
				</div>

				<div className="flex flex-col gap-1">
					<Label className="text-xs text-muted-foreground">Adapter</Label>
					<Select
						value={persona.adapterType}
						onValueChange={(v) =>
							v && onUpdate({ ...persona, adapterType: v as AdapterType })
						}
					>
						<SelectTrigger size="sm" className="w-full">
							<SelectValue />
						</SelectTrigger>
						<SelectPopup>
							<SelectItem value="openai">{ADAPTER_LABELS.openai}</SelectItem>
							<SelectItem value="ollama">{ADAPTER_LABELS.ollama}</SelectItem>
							<SelectItem value="openai-compat">
								{ADAPTER_LABELS["openai-compat"]}
							</SelectItem>
						</SelectPopup>
					</Select>
				</div>
			</div>

			{showEndpoint && (
				<div className="flex flex-col gap-1">
					<Label className="text-xs text-muted-foreground">Endpoint URL</Label>
					<Input
						value={persona.endpointUrl}
						onChange={(e) => onUpdate({ ...persona, endpointUrl: e.target.value })}
						placeholder={
							persona.adapterType === "ollama"
								? "http://localhost:11434"
								: "http://localhost:8000/v1"
						}
						className="h-7 text-xs"
					/>
				</div>
			)}

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
	const { mode, toggle, endpoint, updateEndpoint } = useAssistantMode();

	const handleAdd = () => {
		onAddPersona({
			...DEFAULT_PERSONA,
			id: crypto.randomUUID(),
			scope: "personal",
			name: `Researcher ${session.personas.length + 1}`,
			systemPrompt:
				"You are a specialist researcher. Build on the conversation and add new insights.",
		});
	};

	return (
		<div className="flex flex-col gap-6 p-4 overflow-y-auto">
			<div className="flex flex-col gap-3 rounded-xl border bg-muted/30 p-4">
				<div className="flex items-center justify-between">
					<div className="flex flex-col">
						<span className="text-sm font-medium">Local-only mode</span>
						<span className="text-xs text-muted-foreground">
							Store everything in browser storage. Chat goes to per-persona local endpoints.
						</span>
					</div>
					<Switch
						checked={mode === "local"}
						onCheckedChange={() => toggle()}
					/>
				</div>

				{mode === "local" && (
					<div className="flex flex-col gap-2 pt-2 border-t">
						<div className="flex flex-col gap-1">
							<Label className="text-xs text-muted-foreground">
								Default local endpoint
							</Label>
							<Input
								value={endpoint.baseURL}
								onChange={(e) => updateEndpoint({ baseURL: e.target.value })}
								placeholder="http://localhost:11434"
								className="h-7 text-xs"
							/>
						</div>
						<div className="flex flex-col gap-1">
							<Label className="text-xs text-muted-foreground">
								Auth header (optional)
							</Label>
							<Input
								value={endpoint.authHeader}
								onChange={(e) => updateEndpoint({ authHeader: e.target.value })}
								placeholder="Bearer sk-…"
								className="h-7 text-xs"
							/>
						</div>
					</div>
				)}
			</div>

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
						if (v !== undefined) onWindowSizeChange(v);
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
