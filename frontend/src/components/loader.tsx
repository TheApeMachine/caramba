import { Suspense } from "react";
import { Spinner } from "./ui/spinner";

interface LoaderProps
	extends React.PropsWithChildren<{
		fallback?:
			| React.ReactNode
			| [React.ReactNode, ...React.ReactNode[]]
			| undefined;
	}> {}

export const Loader = ({ children, fallback = <Spinner /> }: LoaderProps) => {
	return <Suspense fallback={fallback}>{children}</Suspense>;
};
