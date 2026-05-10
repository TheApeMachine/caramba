import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface MetricProps {
    label: string;
    value: number;
    icon: React.ReactNode;
}

export const Metric = ({ label, value, icon }: MetricProps) => {
    return (
        <Card className="w-full h-full gap-0">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    {icon}
                    <span className="text-sm font-medium">{label}</span>
                </CardTitle>
            </CardHeader>
            <CardContent>
                <span className="text-2xl font-bold">{value}</span>
            </CardContent>
        </Card>
    );
};
