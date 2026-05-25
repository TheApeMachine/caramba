import { CircleAlertIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import { CardFrameFooter } from "#/components/ui/card";
import { Flex } from "#/components/ui/flex";

export const Footer = () => {
	const { t } = useTranslation();

	return (
		<CardFrameFooter className="shrink-0">
			<Flex.Row gap={1} className="text-muted-foreground text-xs">
				<CircleAlertIcon className="size-3 h-lh shrink-0" />
				<p>{t("assistant.footerStreaming")}</p>
			</Flex.Row>
		</CardFrameFooter>
	);
};
