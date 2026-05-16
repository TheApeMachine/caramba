import { Send, Square } from "lucide-react";
import { Button } from "#/components/ui/button";
import { Field } from "#/components/ui/field";
import { AnimatePresence, Flex } from "#/components/ui/flex";
import { Form } from "#/components/ui/form";
import { Input } from "#/components/ui/input";

interface ComposerProps {
	value: string;
	onChange: (value: string) => void;
	onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
	onStop: () => void;
	busy: boolean;
	placeholder: string;
}

export const Composer = ({
	value,
	onChange,
	onSubmit,
	onStop,
	busy,
	placeholder,
}: ComposerProps) => {
	return (
		<Form onSubmit={onSubmit} className="flex w-full border-t shrink-0">
			<Field className="w-full flex-1">
				<Field.Label className="sr-only">Message</Field.Label>
				<Input
					value={value}
					onChange={(e) => onChange(e.target.value)}
					placeholder={placeholder}
					disabled={busy}
					className="w-full rounded-tr-none rounded-br-none"
				/>
			</Field>
			<AnimatePresence mode="wait" initial={false}>
				<Flex key={busy ? "stop" : "send"} appear="scaleIn">
					<Button
						type={busy ? "button" : "submit"}
						variant={busy ? "outline" : "default"}
						onClick={busy ? onStop : undefined}
						disabled={!busy && !value.trim()}
						className="rounded-tl-none rounded-bl-none"
					>
						{busy ? <Square /> : <Send />}
					</Button>
				</Flex>
			</AnimatePresence>
		</Form>
	);
};
