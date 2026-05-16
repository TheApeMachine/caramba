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
		<Form onSubmit={onSubmit} className="flex gap-2 border-t p-3">
			<Field className="flex-1">
				<Field.Label className="sr-only">Message</Field.Label>
				<Input
					value={value}
					onChange={(e) => onChange(e.target.value)}
					placeholder={placeholder}
					disabled={busy}
				/>
			</Field>
			<AnimatePresence mode="wait" initial={false}>
				{busy ? (
					<Flex key="stop" appear="scaleIn">
						<Button type="button" variant="outline" onClick={onStop}>
							<Square />
						</Button>
					</Flex>
				) : (
					<Flex key="send" appear="scaleIn">
						<Button type="submit" disabled={!value.trim()}>
							<Send />
						</Button>
					</Flex>
				)}
			</AnimatePresence>
		</Form>
	);
};
