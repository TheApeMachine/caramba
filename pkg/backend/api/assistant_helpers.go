package api

import (
	"database/sql"
	"fmt"
	"strings"
	"sync"

	"github.com/gofiber/fiber/v3"
)

/*
clerkIdentity captures the locals the clerk middleware sets. Reading them
through readClerkIdentity makes a missing/wrong-typed value a clear 401
rather than a silent zero-value that bypasses authorization checks.
*/
type clerkIdentity struct {
	subject string
	orgSlug string
	isAdmin bool
}

func readClerkIdentity(ctx fiber.Ctx) (clerkIdentity, error) {
	identity := clerkIdentity{}

	subject, ok := ctx.Locals("clerkSubject").(string)

	if !ok || strings.TrimSpace(subject) == "" {
		return identity, fmt.Errorf("authenticated identity required")
	}

	identity.subject = subject

	if orgSlug, ok := ctx.Locals("clerkOrganizationSlug").(string); ok {
		identity.orgSlug = orgSlug
	}

	if isAdmin, ok := ctx.Locals("clerkAdmin").(bool); ok {
		identity.isAdmin = isAdmin
	}

	return identity, nil
}

/*
runWithTxid wraps a mutation in a transaction, captures pg_current_xact_id so
Electric can reconcile the optimistic write, and returns the txid as int64.
*/
func runWithTxid(
	ctx fiber.Ctx, database *sql.DB, body func(*sql.Tx) error,
) (int64, error) {
	transaction, err := database.BeginTx(ctx.Context(), nil)

	if err != nil {
		return 0, fmt.Errorf("transaction begin: %w", err)
	}

	defer transaction.Rollback()

	if err := body(transaction); err != nil {
		return 0, fmt.Errorf("transaction body: %w", err)
	}

	txidRow := transaction.QueryRowContext(
		ctx.Context(), "SELECT pg_current_xact_id()::xid::text AS txid",
	)

	var txidRaw string

	if err := txidRow.Scan(&txidRaw); err != nil {
		return 0, fmt.Errorf("transaction txid scan: %w", err)
	}

	if err := transaction.Commit(); err != nil {
		return 0, fmt.Errorf("transaction commit: %w", err)
	}

	var txid int64

	if _, err := fmt.Sscan(txidRaw, &txid); err != nil {
		return 0, fmt.Errorf("transaction txid parse: %w", err)
	}

	return txid, nil
}

/*
sqlPool is a lazily-opened, memoized *sql.DB keyed by database URL. Every
service used to carry its own copy of the same five-line open() method; this
collapses that into one shared lazy initializer.
*/
type sqlPool struct {
	url      string
	once     sync.Once
	database *sql.DB
	err      error
}

func newSQLPool(databaseURL string) *sqlPool {
	return &sqlPool{url: strings.TrimSpace(databaseURL)}
}

func (pool *sqlPool) open() (*sql.DB, error) {
	pool.once.Do(func() {
		if pool.url == "" {
			pool.err = fmt.Errorf("database_url is required")
			return
		}

		pool.database, pool.err = sql.Open("postgres", pool.url)
	})

	return pool.database, pool.err
}

/*
mutate is the shared handler shell: bind the JSON payload, require an
authenticated identity, run the supplied operation, and JSON-wrap the txid or
the appropriate error status. Every CRUD handler in this package reduces to
one line that calls mutate with a typed payload and an inline op.
*/
func mutate[T any](
	ctx fiber.Ctx,
	payloadLabel string,
	operation func(fiber.Ctx, clerkIdentity, T) (int64, error),
) error {
	request := *new(T)

	if err := ctx.Bind().JSON(&request); err != nil {
		return ctx.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": fmt.Sprintf("invalid %s payload", payloadLabel)})
	}

	identity, err := readClerkIdentity(ctx)

	if err != nil {
		return ctx.Status(fiber.StatusUnauthorized).JSON(fiber.Map{"error": err.Error()})
	}

	txid, err := operation(ctx, identity, request)

	if err != nil {
		return ctx.Status(statusForError(err)).JSON(fiber.Map{"error": err.Error()})
	}

	return ctx.JSON(fiber.Map{"txid": txid})
}

/*
errForbidden flags a scope/authorization failure so mutate can return 403
instead of 500. Other errors fall through to 500.
*/
type errForbidden struct{ inner error }

func (err errForbidden) Error() string { return err.inner.Error() }

func forbidden(err error) error { return errForbidden{inner: err} }

func statusForError(err error) int {
	if _, ok := err.(errForbidden); ok {
		return fiber.StatusForbidden
	}

	return fiber.StatusInternalServerError
}
