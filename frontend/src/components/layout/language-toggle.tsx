import { LanguagesIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "#/components/ui/button";
import { Menu, MenuItem, MenuPopup, MenuTrigger } from "#/components/ui/menu";
import {
	STORAGE_KEY_LANGUAGE,
	type SupportedLanguage,
	supportedLanguages,
} from "#/i18n";

const languageLabels: Record<SupportedLanguage, string> = {
	en: "language.en",
	de: "language.de",
};

/*
LanguageToggle lets users switch between supported UI languages.
Selection is persisted via i18next-browser-languagedetector localStorage.
*/
export const LanguageToggle = () => {
	const { i18n, t } = useTranslation();
	const activeLanguage = i18n.resolvedLanguage ?? i18n.language;

	return (
		<Menu>
			<MenuTrigger
				render={
					<Button
						aria-label={t("language.toggle")}
						size="icon"
						type="button"
						variant="outline"
					/>
				}
			>
				<LanguagesIcon className="size-4" />
			</MenuTrigger>
			<MenuPopup align="end" className="min-w-40">
				{supportedLanguages.map((languageCode) => (
					<MenuItem
						key={languageCode}
						onClick={() => {
							window.localStorage.setItem(STORAGE_KEY_LANGUAGE, languageCode);
							void i18n.changeLanguage(languageCode);
						}}
					>
						{t(languageLabels[languageCode])}
						{activeLanguage.startsWith(languageCode) ? " ✓" : ""}
					</MenuItem>
				))}
			</MenuPopup>
		</Menu>
	);
};
