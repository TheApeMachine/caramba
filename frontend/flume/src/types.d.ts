declare module "lodash/orderBy" {
    export default function orderBy<T>(
        collection: T[] | Record<string, T>,
        iteratees?:
            | (keyof T | ((item: T) => unknown))[]
            | keyof T
            | ((item: T) => unknown),
        orders?: ("asc" | "desc")[] | "asc" | "desc"
    ): T[];
}
