declare module "*.css" {
    interface IClassNames {
        [className: string]: string;
    }
    const classNames: IClassNames;
    export = classNames;
}

declare module "lodash/clamp" {
    export default function clamp(
        value: number,
        lower?: number,
        upper?: number
    ): number;
}
