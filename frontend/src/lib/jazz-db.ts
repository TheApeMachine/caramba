/*
useJazzDb — a typed view over jazz-tools' useDb().

jazz-tools/react exports useDb<TDb = unknown>(), and does not export its Db
class, so untyped db.insert/update/delete give no payload type-checking. This
wrapper declares the slice of the Db surface we use, inferring each table's
insert/row types from the schema handles (app.table._initType / ._rowType), so
that db.insert(app.kanbanCards, { ... }) is checked against TableInit and the
returned value is typed as the table row.

Signatures mirror jazz-tools@2.0.0-alpha.50 (dist/runtime/db.d.ts):
  insert -> WriteResult<T> ({ value } sync + wait(tier) durability promise)
  update/delete -> WriteHandle (wait(tier))
*/

import { useDb } from "jazz-tools/react";

type DurabilityTier = "local" | "edge" | "global";

type TableHandle<TInit, TRow> = {
	readonly _initType: TInit;
	readonly _rowType: TRow;
};

interface JazzWriteHandle<TRow> {
	wait(options: { tier: DurabilityTier }): Promise<TRow>;
}

interface JazzWriteResult<TRow> extends JazzWriteHandle<TRow> {
	readonly value: TRow;
}

export interface JazzDb {
	insert<TInit, TRow>(
		table: TableHandle<TInit, TRow>,
		data: NoInfer<TInit>,
	): JazzWriteResult<TRow>;
	update<TInit, TRow>(
		table: TableHandle<TInit, TRow>,
		id: string,
		data: Partial<NoInfer<TInit>>,
	): JazzWriteHandle<TRow>;
	delete<TInit, TRow>(
		table: TableHandle<TInit, TRow>,
		id: string,
	): JazzWriteHandle<TRow>;
}

export const useJazzDb = (): JazzDb => useDb<JazzDb>();
