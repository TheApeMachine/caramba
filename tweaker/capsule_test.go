package tweaker

import (
	"sync"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewCapsule(t *testing.T) {
	Convey("Given a new Capsule", t, func() {
		Convey("When created with values", func() {
			capsule := NewCapsule([]string{"a", "b", "c"})

			Convey("It should initialize correctly", func() {
				So(capsule, ShouldNotBeNil)
				So(capsule.idx, ShouldEqual, 0)
				So(capsule.values, ShouldResemble, []string{"a", "b", "c"})
			})
		})

		Convey("When created with empty values", func() {
			capsule := NewCapsule([]string{})

			Convey("It should initialize with empty slice", func() {
				So(capsule, ShouldNotBeNil)
				So(capsule.idx, ShouldEqual, 0)
				So(len(capsule.values), ShouldEqual, 0)
			})
		})
	})
}

func TestNext(t *testing.T) {
	Convey("Given a Capsule with string values", t, func() {
		capsule := NewCapsule([]string{"a", "b", "c"})

		Convey("Next should cycle through values correctly", func() {
			So(capsule.Next(), ShouldEqual, "a")
			So(capsule.Next(), ShouldEqual, "b")
			So(capsule.Next(), ShouldEqual, "c")
			So(capsule.Next(), ShouldEqual, "a") // Should wrap around
		})
	})

	Convey("Given a Capsule with integer values", t, func() {
		capsule := NewCapsule([]int{1, 2, 3})

		Convey("Next should handle numeric types correctly", func() {
			So(capsule.Next(), ShouldEqual, 1)
			So(capsule.Next(), ShouldEqual, 2)
			So(capsule.Next(), ShouldEqual, 3)
			So(capsule.Next(), ShouldEqual, 1)
		})
	})

	Convey("Given an empty Capsule", t, func() {
		capsule := NewCapsule([]string{})

		Convey("Next should return zero value", func() {
			So(capsule.Next(), ShouldEqual, "")
		})
	})
}

func TestPeek(t *testing.T) {
	Convey("Given a Capsule with values", t, func() {
		capsule := NewCapsule([]string{"a", "b", "c"})

		Convey("Peek should return current value without advancing", func() {
			So(capsule.Peek(), ShouldEqual, "a")
			So(capsule.Peek(), ShouldEqual, "a")
			capsule.Next()
			So(capsule.Peek(), ShouldEqual, "b")
		})
	})

	Convey("Given an empty Capsule", t, func() {
		capsule := NewCapsule([]string{})

		Convey("Peek should return zero value", func() {
			So(capsule.Peek(), ShouldEqual, "")
		})
	})
}

func TestReset(t *testing.T) {
	Convey("Given a Capsule with values", t, func() {
		capsule := NewCapsule([]string{"a", "b", "c"})

		Convey("Reset should return to the beginning", func() {
			capsule.Next() // a
			capsule.Next() // b
			capsule.Reset()
			So(capsule.Next(), ShouldEqual, "a")
		})
	})
}

func TestConcurrentAccess(t *testing.T) {
	Convey("Given a Capsule accessed concurrently", t, func() {
		capsule := NewCapsule([]int{1, 2, 3})
		iterations := 100

		Convey("It should handle concurrent Next calls safely", func() {
			var wg sync.WaitGroup
			results := make([]int, iterations)

			for i := 0; i < iterations; i++ {
				wg.Add(1)
				go func(idx int) {
					defer wg.Done()
					results[idx] = capsule.Next()
				}(i)
			}
			wg.Wait()

			validValues := map[int]bool{1: true, 2: true, 3: true}
			for _, v := range results {
				So(validValues[v], ShouldBeTrue)
			}
		})

		Convey("It should handle concurrent mixed operations safely", func() {
			var wg sync.WaitGroup
			ops := make(chan func(), iterations)
			results := make(chan int, iterations)

			// Queue up mixed operations
			for i := 0; i < iterations; i++ {
				wg.Add(1)
				switch i % 3 {
				case 0:
					ops <- func() { results <- capsule.Next() }
				case 1:
					ops <- func() { results <- capsule.Peek() }
				case 2:
					ops <- func() { capsule.Reset(); results <- capsule.Next() }
				}
			}

			// Execute operations concurrently
			for i := 0; i < iterations; i++ {
				go func() {
					defer wg.Done()
					op := <-ops
					op()
				}()
			}

			wg.Wait()
			close(results)

			validValues := map[int]bool{1: true, 2: true, 3: true}
			count := 0
			for v := range results {
				So(validValues[v], ShouldBeTrue)
				count++
			}
			So(count, ShouldEqual, iterations)
		})
	})
}

func TestCustomTypes(t *testing.T) {
	Convey("Given a Capsule with custom types", t, func() {
		type CustomType struct {
			value string
		}
		capsule := NewCapsule([]CustomType{
			{value: "first"},
			{value: "second"},
		})

		Convey("It should handle custom types correctly", func() {
			first := capsule.Next()
			So(first.value, ShouldEqual, "first")
			second := capsule.Next()
			So(second.value, ShouldEqual, "second")
		})
	})
}

func TestIter(t *testing.T) {
	Convey("Given a Capsule with string values", t, func() {
		capsule := NewCapsule([]string{"one", "two", "three"})

		Convey("Iterator should match Next behavior", func() {
			var sequence []string
			iter := capsule.Iter()
			done := false
			iter(func(v string) bool {
				sequence = append(sequence, v)
				done = len(sequence) >= 3
				return !done
			})
			So(sequence, ShouldResemble, []string{"one", "two", "three"})
		})

		Convey("Iterator should respect early termination", func() {
			var sequence []string
			iter := capsule.Iter()
			done := false
			iter(func(v string) bool {
				sequence = append(sequence, v)
				done = len(sequence) >= 2
				return !done
			})
			So(len(sequence), ShouldEqual, 2)
		})
	})

	Convey("Given an empty Capsule", t, func() {
		capsule := NewCapsule([]string{})

		Convey("Iterator should return zero values", func() {
			count := 0
			iter := capsule.Iter()
			iter(func(v string) bool {
				count++
				return false // Stop after first value
			})
			So(count, ShouldEqual, 1)
		})
	})

	Convey("Given a Capsule accessed concurrently through iterator", t, func() {
		capsule := NewCapsule([]int{1, 2, 3})
		var wg sync.WaitGroup
		numGoroutines := 10

		Convey("Multiple goroutines should get consistent sequences", func() {
			results := make([][]int, numGoroutines)
			for i := 0; i < numGoroutines; i++ {
				wg.Add(1)
				go func(idx int) {
					defer wg.Done()
					var sequence []int
					iter := capsule.Iter()
					done := false
					iter(func(v int) bool {
						sequence = append(sequence, v)
						done = len(sequence) >= 3
						return !done
					})
					results[idx] = sequence
				}(i)
			}
			wg.Wait()

			// Verify each goroutine got all values
			for _, sequence := range results {
				So(sequence, ShouldContain, 1)
				So(sequence, ShouldContain, 2)
				So(sequence, ShouldContain, 3)
				So(len(sequence), ShouldEqual, 3)
			}
		})
	})
}
