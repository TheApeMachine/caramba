package device

import (
	"fmt"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ir"
)

func TestEnumerateBackendMethods(t *testing.T) {
	Convey("Given device.Backend embedded interfaces", t, func() {
		entries := EnumerateBackendMethods()

		Convey("It should enumerate 151 methods across 25 surfaces", func() {
			So(len(entries), ShouldEqual, 151)

			surfaceCounts := make(map[string]int)

			for _, entry := range entries {
				surfaceCounts[entry.Surface]++
			}

			So(surfaceCounts["Activation"], ShouldEqual, 55)
			So(surfaceCounts["PosPop"], ShouldEqual, 5)
			So(len(surfaceCounts), ShouldEqual, 25)
		})

		Convey("It should have unique surface.method pairs", func() {
			seen := make(map[BackendMethodEntry]bool, len(entries))

			for _, entry := range entries {
				So(seen[entry], ShouldBeFalse)

				seen[entry] = true
			}
		})
	})
}

func TestValidateBackendInventory(t *testing.T) {
	Convey("Given the Backend ↔ IR cross-link inventory", t, func() {
		Convey("It should validate every required operation ID", func() {
			So(ValidateBackendInventory(), ShouldBeNil)
		})
	})
}

func TestBuildOperationCrossLinkIndex(t *testing.T) {
	Convey("Given ir.RequiredOperationIDs", t, func() {
		index := BuildOperationCrossLinkIndex()
		requiredIDs := ir.RequiredOperationIDs()

		Convey("It should index every required ID exactly once", func() {
			So(len(index), ShouldEqual, len(requiredIDs))

			for _, operationID := range requiredIDs {
				_, ok := index[operationID]

				So(ok, ShouldBeTrue)
			}
		})

		Convey("It should classify link kinds with expected counts", func() {
			kindCounts := make(map[CrossLinkKind]int)

			for _, link := range index {
				kindCounts[link.Kind]++
			}

			So(kindCounts[CrossLinkDirect], ShouldEqual, 74)
			So(kindCounts[CrossLinkComposite], ShouldEqual, 3)
			So(kindCounts[CrossLinkKernelRegistry], ShouldEqual, 32)
			So(kindCounts[CrossLinkGraphOnly], ShouldEqual, 10)
		})
	})
}

func TestBackendMethodsWithoutRequiredOperation(t *testing.T) {
	Convey("Given Backend methods not referenced by required operation IDs", t, func() {
		unmapped := BackendMethodsWithoutRequiredOperation()

		Convey("It should list 88 backend-only methods", func() {
			So(len(unmapped), ShouldEqual, 88)
		})
	})
}

func BenchmarkBuildOperationCrossLinkIndex(b *testing.B) {
	for b.Loop() {
		_ = BuildOperationCrossLinkIndex()
	}
}

func BenchmarkValidateBackendInventory(b *testing.B) {
	for b.Loop() {
		if err := ValidateBackendInventory(); err != nil {
			b.Fatal(err)
		}
	}
}

func ExampleValidateBackendInventory() {
	if err := ValidateBackendInventory(); err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("inventory ok")
	// Output: inventory ok
}
