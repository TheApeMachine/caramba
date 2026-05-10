/**
 * Color spreading calculation: cycles through chart colors 1-10
 */
export function getColorIndex(index: number): number {
    return (index % 10) + 1;
}
