package executor

import (
	"context"
	"errors"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

type recorderOperation struct {
	executions *[]string
}

func (recorder recorderOperation) Execute(execContext op.Context) error {
	*recorder.executions = append(*recorder.executions, execContext.Step().ID)

	for outputName, ref := range execContext.Step().Outputs {
		if err := execContext.Bind(ref, outputName); err != nil {
			return err
		}
	}

	return nil
}

type bodyRunnerOperation struct{}

func (bodyRunnerOperation) Execute(execContext op.Context) error {
	return execContext.Run(execContext.Step().Body)
}

type breakingOperation struct{}

func (breakingOperation) Execute(execContext op.Context) error {
	return op.ErrBreak
}

type loopOperation struct {
	maxIterations int
}

func (loop loopOperation) Execute(execContext op.Context) error {
	for index := 0; index < loop.maxIterations; index++ {
		err := execContext.Run(execContext.Step().Body)

		if errors.Is(err, op.ErrBreak) {
			return nil
		}

		if errors.Is(err, op.ErrContinue) {
			continue
		}

		if err != nil {
			return err
		}
	}

	return nil
}

func newRegistry(executions *[]string) *op.Registry {
	registry := op.NewRegistry()
	registry.MustRegister("test.record", recorderOperation{executions: executions})
	registry.MustRegister("test.body", bodyRunnerOperation{})
	registry.MustRegister("test.break", breakingOperation{})
	registry.MustRegister("test.loop3", loopOperation{maxIterations: 3})

	return registry
}

func TestExecutorRun(t *testing.T) {
	Convey("Given an executor with a small program", t, func() {
		executions := []string{}
		registry := newRegistry(&executions)

		runtimeProgram := &program.Program{
			Name: "test",
			Steps: []program.Step{
				{
					ID: "first",
					Op: "test.record",
					Outputs: map[string]ValueRefAlias{
						"primary": {Namespace: program.NamespaceLocal, Name: "primary"},
					},
				},
				{ID: "second", Op: "test.record"},
			},
		}

		executor, err := New(Options{Program: runtimeProgram, Operations: registry})
		So(err, ShouldBeNil)

		Convey("Run should dispatch steps in order", func() {
			So(executor.Run(context.Background()), ShouldBeNil)
			So(executions, ShouldResemble, []string{"first", "second"})
		})

		Convey("Run should surface unknown-op errors with step id and op", func() {
			runtimeProgram.Steps = append(runtimeProgram.Steps, program.Step{
				ID: "third",
				Op: "test.missing",
			})
			executor, err := New(Options{Program: runtimeProgram, Operations: registry})
			So(err, ShouldBeNil)

			err = executor.Run(context.Background())
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "third")
			So(err.Error(), ShouldContainSubstring, "test.missing")
		})
	})

	Convey("Given a program with a nested body", t, func() {
		executions := []string{}
		registry := newRegistry(&executions)

		runtimeProgram := &program.Program{
			Name: "nested",
			Steps: []program.Step{
				{
					ID: "outer",
					Op: "test.body",
					Body: []program.Step{
						{ID: "inner_a", Op: "test.record"},
						{ID: "inner_b", Op: "test.record"},
					},
				},
			},
		}

		executor, err := New(Options{Program: runtimeProgram, Operations: registry})
		So(err, ShouldBeNil)

		Convey("Nested steps should run when the parent invokes Run", func() {
			So(executor.Run(context.Background()), ShouldBeNil)
			So(executions, ShouldResemble, []string{"inner_a", "inner_b"})
		})
	})

	Convey("Given a loop with a break op in its body", t, func() {
		executions := []string{}
		registry := newRegistry(&executions)

		runtimeProgram := &program.Program{
			Name: "loop_break",
			Steps: []program.Step{
				{
					ID: "loop",
					Op: "test.loop3",
					Body: []program.Step{
						{ID: "before_break", Op: "test.record"},
						{ID: "stop", Op: "test.break"},
						{ID: "after_break", Op: "test.record"},
					},
				},
			},
		}

		executor, err := New(Options{Program: runtimeProgram, Operations: registry})
		So(err, ShouldBeNil)

		Convey("Break should exit the loop after the first iteration", func() {
			So(executor.Run(context.Background()), ShouldBeNil)
			So(executions, ShouldResemble, []string{"before_break"})
		})
	})

	Convey("Given a context that has been cancelled", t, func() {
		executions := []string{}
		registry := newRegistry(&executions)

		runtimeProgram := &program.Program{
			Name: "cancelled",
			Steps: []program.Step{
				{ID: "first", Op: "test.record"},
				{ID: "second", Op: "test.record"},
			},
		}

		executor, err := New(Options{Program: runtimeProgram, Operations: registry})
		So(err, ShouldBeNil)

		Convey("Run should stop at the first step and return ctx.Err", func() {
			cancelCtx, cancel := context.WithCancel(context.Background())
			cancel()

			err := executor.Run(cancelCtx)
			So(err, ShouldNotBeNil)
			So(errors.Is(err, context.Canceled), ShouldBeTrue)
			So(executions, ShouldBeEmpty)
		})
	})

	Convey("New should reject programs that fail validation", t, func() {
		runtimeProgram := &program.Program{Name: ""}
		_, err := New(Options{Program: runtimeProgram, Operations: op.NewRegistry()})
		So(err, ShouldNotBeNil)
	})
}

// ValueRefAlias keeps test struct literals readable.
type ValueRefAlias = program.ValueRef
