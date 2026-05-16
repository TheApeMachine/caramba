import { CircleAlertIcon } from "lucide-react";
import { CardFrameFooter } from "#/components/ui/card";
import { Flex } from "#/components/ui/flex";

export const Footer = () => {
	return (
		<CardFrameFooter>
			<Flex.Row gap={1} className="text-muted-foreground text-xs">
				<CircleAlertIcon className="size-3 h-lh shrink-0" />
				<p>Responses may be incomplete while the team is streaming.</p>
			</Flex.Row>
		</CardFrameFooter>
	);
};
