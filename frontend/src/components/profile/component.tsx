import type { useUser } from "@clerk/tanstack-react-start";
import { Avatar, AvatarFallback, AvatarImage } from "#/components/ui/avatar";
import { Flex } from "#/components/ui/flex";

interface ProfileComponentProps {
	user: NonNullable<ReturnType<typeof useUser>["user"]>;
}

export const ProfileComponent = ({ user }: ProfileComponentProps) => {
	return (
		<Flex.Column gap={3} padding={2} fullWidth>
			<Flex.Row align="start" gap={3} fullWidth>
				<Avatar className="size-14 border-2 border-primary/20">
					{user.imageUrl ? <AvatarImage alt="" src={user.imageUrl} /> : null}
					<AvatarFallback className="text-base">
						{user.fullName?.slice(0, 2).toUpperCase() ?? "?"}
					</AvatarFallback>
				</Avatar>
			</Flex.Row>
		</Flex.Column>
	);
};
