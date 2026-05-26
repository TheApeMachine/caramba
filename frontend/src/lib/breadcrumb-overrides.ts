import { Store, useStore } from "@tanstack/react-store";
import { useEffect } from "react";

/*
BreadcrumbOverrides is keyed by canonical href so the breadcrumb can
swap in human-readable labels for slug or UUID segments without the
shell knowing anything about the underlying entities.
*/
type BreadcrumbOverrides = Record<string, string>;

const breadcrumbOverridesStore = new Store<BreadcrumbOverrides>({});

const setBreadcrumbOverride = (href: string, label: string | null) => {
	breadcrumbOverridesStore.setState((previous) => {
		if (label === null) {
			if (!(href in previous)) {
				return previous;
			}

			const next = { ...previous };
			delete next[href];
			return next;
		}

		if (previous[href] === label) {
			return previous;
		}

		return { ...previous, [href]: label };
	});
};

export const useBreadcrumbOverrides = () => useStore(breadcrumbOverridesStore);

/*
useBreadcrumbOverride registers a label override for the given href
while the calling component is mounted. Pass null/undefined when the
label is not yet known so the override is automatically cleared once
the component unmounts or the label resolves later.
*/
export const useBreadcrumbOverride = (
	href: string,
	label: string | null | undefined,
) => {
	useEffect(() => {
		if (!label) {
			return;
		}

		setBreadcrumbOverride(href, label);

		return () => {
			setBreadcrumbOverride(href, null);
		};
	}, [href, label]);
};
