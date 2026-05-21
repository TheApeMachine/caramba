package apidata

import (
	"database/sql"
	"fmt"
	"strings"
	"sync"

	"github.com/gofiber/fiber/v3"
	_ "github.com/lib/pq"
)

/*
ClerkIdentity captures the locals the clerk middleware sets. Reading them
through ReadClerkIdentity makes a missing/wrong-typed value a clear 401
rather than a silent zero-value that bypasses authorization checks.
*/
type ClerkIdentity struct {
	Subject string
	OrgSlug string
	IsAdmin bool
}

func ReadClerkIdentity(ctx fiber.Ctx) (ClerkIdentity, error) {
	identity := ClerkIdentity{}

	subject, ok := ctx.Locals("clerkSubject").(string)

	if !ok || strings.TrimSpace(subject) == "" {
		return identity, fmt.Errorf("authenticated identity required")
	}

	identity.Subject = subject

	if orgSlug, ok := ctx.Locals("clerkOrganizationSlug").(string); ok {
		identity.OrgSlug = orgSlug
	}

	if isAdmin, ok := ctx.Locals("clerkAdmin").(bool); ok {
		identity.IsAdmin = isAdmin
	}

	return identity, nil
}

/*
RunWithTxid wraps a mutation in a transaction, captures pg_current_xact_id so
Electric can reconcile the optimistic write, and returns the txid as int64.
*/
func RunWithTxid(
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
SQLPool is a lazily-opened, memoized *sql.DB keyed by database URL.
*/
type SQLPool struct {
	url      string
	once     sync.Once
	database *sql.DB
	err      error
}

func NewSQLPool(databaseURL string) *SQLPool {
	return &SQLPool{url: strings.TrimSpace(databaseURL)}
}

func (pool *SQLPool) Open() (*sql.DB, error) {
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
Mutate is the shared handler shell: bind the JSON payload, require an
authenticated identity, run the supplied operation, and JSON-wrap the txid or
the appropriate error status.
*/
func Mutate[T any](
	ctx fiber.Ctx,
	payloadLabel string,
	operation func(fiber.Ctx, ClerkIdentity, T) (int64, error),
) error {
	request := *new(T)

	if err := ctx.Bind().JSON(&request); err != nil {
		return ctx.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": fmt.Sprintf("invalid %s payload", payloadLabel)})
	}

	identity, err := ReadClerkIdentity(ctx)

	if err != nil {
		return ctx.Status(fiber.StatusUnauthorized).JSON(fiber.Map{"error": err.Error()})
	}

	txid, err := operation(ctx, identity, request)

	if err != nil {
		return ctx.Status(statusForError(err)).JSON(fiber.Map{"error": err.Error()})
	}

	return ctx.JSON(fiber.Map{"txid": txid})
}

type errForbidden struct{ inner error }

func (err errForbidden) Error() string { return err.inner.Error() }

func Forbidden(err error) error { return errForbidden{inner: err} }

func statusForError(err error) int {
	if _, ok := err.(errForbidden); ok {
		return fiber.StatusForbidden
	}

	return fiber.StatusInternalServerError
}

func NullString(value string) sql.NullString {
	return sql.NullString{String: value, Valid: value != ""}
}
