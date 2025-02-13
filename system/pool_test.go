package system

import (
	"strconv"
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tests"
	"github.com/theapemachine/caramba/tools"
)

// TestNewPool tests the singleton pattern of Pool creation
func TestNewPool(t *testing.T) {
	Convey("Given a new pool", t, func() {
		pool := NewPool()
		So(pool, ShouldNotBeNil)

		Convey("When called multiple times", func() {
			pool2 := NewPool()
			So(pool2, ShouldNotBeNil)

			Convey("It should return the same pool instance", func() {
				So(pool2, ShouldEqual, pool)
			})
		})
	})
}

func TestPoolAdd(t *testing.T) {
	Convey("Given a new pool", t, func() {
		pool := NewPool()

		Convey("When adding an entity", func() {
			pool.Add(
				"system prompt",
				"role",
				"name",
				tools.NewToolset(),
			)

			Convey("It should add the entity to the pool", func() {
				So(pool.entities["name"], ShouldNotBeNil)

				Convey("The entity should have the correct properties", func() {
					So(pool.entities["name"].Config.SystemPrompt, ShouldEqual, "system prompt")
					So(pool.entities["name"].Config.Role, ShouldEqual, "role")
					So(pool.entities["name"].Config.Name, ShouldEqual, "name")
				})
			})
		})
	})
}

func TestPoolSelect(t *testing.T) {
	Convey("Given a new pool", t, func() {
		pool := NewPool()

		Convey("When multiple entities are added", func() {
			idx := 0

			for range 3 {
				pool.Add(
					"system prompt",
					"role",
					"name"+strconv.Itoa(idx),
					tools.NewToolset(),
				)
			}

			Convey("When selecting an entity", func() {
				pool.Select("name1")

				Convey("It should return the selected entity", func() {
					So(pool.selected, ShouldEqual, pool.entities["name1"])
				})

				pool.Select("name2")

				Convey("It should return the next selected entity", func() {
					So(pool.selected, ShouldEqual, pool.entities["name2"])
				})

				Convey("When selecting a non-existent entity", func() {
					pool.Select("non-existent")

					Convey("It should return nil", func() {
						So(pool.selected, ShouldBeNil)
					})
				})
			})
		})
	})
}

func TestPoolGenerate(t *testing.T) {
	Convey("Given a new pool", t, func() {
		pool := NewPool()

		Convey("And the pool has an entity", func() {
			mock := tests.NewMockGenerator()

			pool.entities["test"] = &Entity{
				Config:    mock.Ctx().Config(),
				Generator: mock,
				Toolset:   tools.NewToolset(),
			}

			Convey("When the entity is selected", func() {
				pool.Select("test")

				Convey("When generating a message", func() {
					var catcher strings.Builder

					for event := range pool.Generate(provider.NewMessage(
						provider.RoleUser,
						"Hello, world!",
					)) {
						if event.Type == provider.EventChunk {
							catcher.WriteString(event.Text)
						}
					}

					Convey("It should return a message", func() {
						So(pool.out, ShouldNotBeNil)
						So(catcher.String(), ShouldEqual, "Hello, world!")
					})
				})

				Convey("When accumulating the output", func() {
					accumulator := stream.NewAccumulator()

					for event := range accumulator.Generate(pool.Generate(provider.NewMessage(
						provider.RoleUser,
						"Hello, world!",
					))) {
						_ = event
					}

					Convey("It should return the accumulated output", func() {
						So(accumulator.String(), ShouldEqual, "Hello, world!")
					})
				})
			})
		})
	})
}
