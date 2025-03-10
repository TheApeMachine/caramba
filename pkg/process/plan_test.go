package process

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// TestNewPlan tests the initialization of Plan
func TestNewPlan(t *testing.T) {
	Convey("Given the need for a new Plan", t, func() {
		Convey("When creating a new Plan", func() {
			plan := NewPlan()

			Convey("Then it should be properly initialized", func() {
				So(plan, ShouldNotBeNil)
				So(plan.PlanData, ShouldNotBeNil)
				So(len(plan.PlanData.Steps), ShouldEqual, 0)
				So(plan.in, ShouldNotBeNil)
				So(plan.out, ShouldNotBeNil)
				So(plan.enc, ShouldNotBeNil)
				So(plan.dec, ShouldNotBeNil)
			})
		})
	})
}

// TestPlanWithSteps tests adding steps to a Plan
func TestPlanWithSteps(t *testing.T) {
	Convey("Given a Plan", t, func() {
		plan := NewPlan()

		Convey("When adding steps", func() {
			step1 := Step{Step: "Step 1"}
			step2 := Step{Step: "Step 2"}

			resultPlan := plan.WithSteps(step1, step2)

			Convey("Then the steps should be added", func() {
				So(resultPlan, ShouldEqual, plan) // Should return the same instance
				So(len(plan.PlanData.Steps), ShouldEqual, 2)
				So(plan.PlanData.Steps[0].Step, ShouldEqual, "Step 1")
				So(plan.PlanData.Steps[1].Step, ShouldEqual, "Step 2")
			})

			Convey("And when adding more steps", func() {
				step3 := Step{Step: "Step 3"}
				resultPlan = plan.WithSteps(step3)

				Convey("Then they should be appended", func() {
					So(resultPlan, ShouldEqual, plan)
					So(len(plan.PlanData.Steps), ShouldEqual, 3)
					So(plan.PlanData.Steps[2].Step, ShouldEqual, "Step 3")
				})
			})
		})
	})
}

// TestPlanReadBasics tests the read functionality of the Plan
func TestPlanReadBasics(t *testing.T) {
	Convey("Given a Plan with steps", t, func() {
		plan := NewPlan().WithSteps(
			Step{Step: "Step 1"},
			Step{Step: "Step 2"},
		)

		Convey("When reading from the plan", func() {
			// The NewPlan constructor and WithSteps already encoded the data to the output buffer
			// so we don't need to manually re-encode here

			buf := make([]byte, 1024)
			n, err := plan.Read(buf)

			Convey("Then reading should succeed", func() {
				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)

				// Now we can check for specific step content since WithSteps re-encodes the data
				jsonStr := string(buf[:n])
				So(jsonStr, ShouldContainSubstring, "Step 1")
				So(jsonStr, ShouldContainSubstring, "Step 2")
			})
		})
	})
}

// TestPlanWriteBasics tests the basic IO functionality of Write
func TestPlanWriteBasics(t *testing.T) {
	Convey("Given a Plan", t, func() {
		plan := NewPlan()

		Convey("When writing to the plan's buffer directly", func() {
			testBytes := []byte(`{"steps":[{"step":"Test Step"}]}`)
			n, err := plan.in.Write(testBytes)

			Convey("Then it should accept the data correctly", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(testBytes))
				So(plan.in.Len(), ShouldBeGreaterThan, 0)
			})
		})
	})
}

// TestPlanClose tests the Close method
func TestPlanClose(t *testing.T) {
	Convey("Given a Plan", t, func() {
		plan := NewPlan().WithSteps(Step{Step: "Test step"})

		Convey("When closing the plan", func() {
			err := plan.Close()

			Convey("Then it should close successfully", func() {
				So(err, ShouldBeNil)
				So(len(plan.PlanData.Steps), ShouldEqual, 0) // Steps should be cleared
			})
		})
	})
}
