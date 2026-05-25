import { useTranslation } from "react-i18next";

export const NotFoundPage = () => {
	const { t } = useTranslation();

	return (
		<div className="flex h-full items-center justify-center text-muted-foreground">
			{t("errors.pageNotFound")}
		</div>
	);
};
