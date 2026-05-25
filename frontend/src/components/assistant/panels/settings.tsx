import { Plus, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
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
import {
	type AdapterType,
	AVAILABLE_MODELS,
	DEFAULT_PERSONA,
	type Persona,
	type PersonaScope,
	type Provider,
	type Session,
} from "../types";
import { useAssistantMode } from "../use-assistant-mode";

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
	global: "assistant.scope.global",
	team: "assistant.scope.team",
	personal: "assistant.scope.personal",
};

const ADAPTER_LABELS: Record<AdapterType, string> = {
	openai: "assistant.adapter.openai",
	ollama: "assistant.adapter.ollama",
	"openai-compat": "assistant.adapter.openaiCompat",
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
	const { t } = useTranslation();
	const showEndpoint =
		persona.adapterType === "ollama" || persona.adapterType === "openai-compat";

	return (
		<div className="rounded-xl border bg-muted/30 p-4 flex flex-col gap-3">
			<div className="flex items-center gap-2">
				<Input
					value={persona.name}
					onChange={(e) => onUpdate({ ...persona, name: e.target.value })}
					className="h-7 text-sm font-medium flex-1"
					placeholder={t("assistant.settings.personaNamePlaceholder")}
				/>
				{removable && (
					<Button size="icon-xs" variant="ghost" onClick={onRemove}>
						<Trash2 />
					</Button>
				)}
			</div>

			<div className="grid grid-cols-2 gap-2">
				<div className="flex flex-col gap-1">
					<Label className="text-xs text-muted-foreground">
						{t("assistant.settings.scope")}
					</Label>
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
							<SelectItem value="personal">
								{t(SCOPE_LABELS.personal)}
							</SelectItem>
							<SelectItem value="team">{t(SCOPE_LABELS.team)}</SelectItem>
							<SelectItem value="global">{t(SCOPE_LABELS.global)}</SelectItem>
						</SelectPopup>
					</Select>
				</div>

				<div className="flex flex-col gap-1">
					<Label className="text-xs text-muted-foreground">
						{t("assistant.settings.adapter")}
					</Label>
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
							<SelectItem value="openai">{t(ADAPTER_LABELS.openai)}</SelectItem>
							<SelectItem value="ollama">{t(ADAPTER_LABELS.ollama)}</SelectItem>
							<SelectItem value="openai-compat">
								{t(ADAPTER_LABELS["openai-compat"])}
							</SelectItem>
						</SelectPopup>
					</Select>
				</div>
			</div>

			{showEndpoint && (
				<div className="flex flex-col gap-1">
					<Label className="text-xs text-muted-foreground">
						{t("assistant.settings.endpointUrl")}
					</Label>
					<Input
						value={persona.endpointUrl}
						onChange={(e) =>
							onUpdate({ ...persona, endpointUrl: e.target.value })
						}
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
				placeholder={t("assistant.settings.systemPromptPlaceholder")}
				className="text-xs min-h-[72px]"
			/>

			<div className="flex flex-col gap-2">
				<Label className="text-xs text-muted-foreground">
					{t("assistant.settings.model")}
				</Label>
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
								{AVAILABLE_MODELS.filter(
									(modelEntry) => modelEntry.provider === provider,
								).map((modelEntry) => (
									<SelectItem key={modelEntry.id} value={modelEntry.id}>
										{t(`assistant.models.${modelEntry.id}`, {
											defaultValue: modelEntry.label,
										})}
									</SelectItem>
								))}
							</div>
						))}
					</SelectPopup>
				</Select>
			</div>

			<div className="flex flex-col gap-1">
				<div className="flex items-center justify-between">
					<Label className="text-xs text-muted-foreground">
						{t("assistant.settings.temperature")}
					</Label>
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
					<Label className="text-xs text-muted-foreground">
						{t("assistant.settings.maxTokens")}
					</Label>
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
	const { t } = useTranslation();

	const handleAdd = () => {
		onAddPersona({
			...DEFAULT_PERSONA,
			id: crypto.randomUUID(),
			scope: "personal",
			name: t("assistant.settings.newPersonaName", {
				count: session.personas.length + 1,
			}),
			systemPrompt: t("assistant.settings.defaultSystemPrompt"),
		});
	};

	return (
		<div className="flex flex-col gap-6 p-4 overflow-y-auto">
			<div className="flex flex-col gap-3 rounded-xl border bg-muted/30 p-4">
				<div className="flex items-center justify-between">
					<div className="flex flex-col">
						<span className="text-sm font-medium">
							{t("assistant.settings.localOnlyMode")}
						</span>
						<span className="text-xs text-muted-foreground">
							{t("assistant.settings.localOnlyDescription")}
						</span>
					</div>
					<Switch checked={mode === "local"} onCheckedChange={() => toggle()} />
				</div>

				{mode === "local" && (
					<div className="flex flex-col gap-2 pt-2 border-t">
						<div className="flex flex-col gap-1">
							<Label className="text-xs text-muted-foreground">
								{t("assistant.settings.defaultLocalEndpoint")}
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
								{t("assistant.settings.authHeaderOptional")}
							</Label>
							<Input
								value={endpoint.authHeader}
								onChange={(e) => updateEndpoint({ authHeader: e.target.value })}
								placeholder={t("assistant.settings.authHeaderPlaceholder")}
								className="h-7 text-xs"
							/>
						</div>
					</div>
				)}
			</div>

			<div className="flex flex-col gap-3">
				<div className="flex items-center justify-between">
					<span className="text-sm font-medium">
						{t("assistant.settings.personas")}
					</span>
					<Button size="xs" variant="outline" onClick={handleAdd}>
						<Plus />
						{t("assistant.settings.addPersona")}
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
						{t("assistant.settings.contextWindow")}
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
					{t("assistant.settings.contextWindowHint")}
				</p>
			</div>
		</div>
	);
}
