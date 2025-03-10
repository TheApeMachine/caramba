package core

import (
	"encoding/json"
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// MockFunction implements ReadWriteCloser for testing
type MockFunction struct {
	readData  []byte
	writeData []byte
	closed    bool
}

func NewMockFunction(data string) *MockFunction {
	return &MockFunction{
		readData:  []byte(data),
		writeData: []byte{},
		closed:    false,
	}
}

func (m *MockFunction) Read(p []byte) (n int, err error) {
	if len(m.readData) == 0 {
		return 0, io.EOF
	}

	n = copy(p, m.readData)
	m.readData = m.readData[n:]
	return n, nil
}

func (m *MockFunction) Write(p []byte) (n int, err error) {
	m.writeData = append(m.writeData, p...)
	return len(p), nil
}

func (m *MockFunction) Close() error {
	m.closed = true
	return nil
}

func (m *MockFunction) WasClosed() bool {
	return m.closed
}

func (m *MockFunction) GetWrittenData() []byte {
	return m.writeData
}

// TestNewTool tests the NewTool constructor
func TestNewTool(t *testing.T) {
	Convey("Given parameters for a new Tool", t, func() {
		name := "test_tool"
		description := "A test tool"
		parameters := []Parameter{
			{
				Type: "object",
				Properties: map[string]Property{
					"param1": {
						Type:        "string",
						Description: "A string parameter",
					},
				},
				Required: []string{"param1"},
			},
		}

		Convey("When creating a new Tool", func() {
			tool := NewTool(name, description, parameters)

			Convey("Then the tool should have the correct properties", func() {
				So(tool, ShouldNotBeNil)
				So(tool.Name, ShouldEqual, name)
				So(tool.Description, ShouldEqual, description)
				So(len(tool.Parameters), ShouldEqual, 1)
				So(tool.Parameters[0].Type, ShouldEqual, "object")
				So(tool.Parameters[0].Required[0], ShouldEqual, "param1")
				So(tool.Strict, ShouldBeTrue)

				// Verify that buffers are initialized
				So(tool.in, ShouldNotBeNil)
				So(tool.out, ShouldNotBeNil)
				So(tool.enc, ShouldNotBeNil)
				So(tool.dec, ShouldNotBeNil)
				So(tool.fn, ShouldBeNil) // Function should be nil initially
			})
		})
	})
}

// TestToolRead tests the Read method of Tool
func TestToolRead(t *testing.T) {
	Convey("Given a Tool", t, func() {
		tool := NewTool("test_tool", "Test description", []Parameter{})
		buffer := make([]byte, 1024)

		Convey("When reading from the tool", func() {
			n, err := tool.Read(buffer)

			Convey("Then it should return tool data as JSON", func() {
				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)

				var parsed ToolData
				err := json.Unmarshal(buffer[:n], &parsed)
				So(err, ShouldBeNil)
				So(parsed.Name, ShouldEqual, "test_tool")
				So(parsed.Description, ShouldEqual, "Test description")
			})
		})

		Convey("When reading until empty", func() {
			// First read to consume the buffer
			firstBuffer := make([]byte, 1024)
			tool.Read(firstBuffer)

			// The tool's Read method should re-encode if buffer is empty
			n, err := tool.Read(buffer)

			Convey("Then it should still return data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)
			})
		})
	})
}

// TestToolWrite tests the Write method of Tool
func TestToolWrite(t *testing.T) {
	Convey("Given a Tool", t, func() {
		tool := NewTool("original_name", "Original description", []Parameter{})

		Convey("When writing valid JSON data", func() {
			updatedData := &ToolData{
				Name:        "updated_name",
				Description: "Updated description",
				Parameters: []Parameter{
					{
						Type: "string",
						Properties: map[string]Property{
							"new_param": {
								Type:        "string",
								Description: "A new parameter",
							},
						},
					},
				},
				Strict: false,
			}

			jsonData, err := json.Marshal(updatedData)
			So(err, ShouldBeNil)

			n, err := tool.Write(jsonData)

			Convey("Then it should update the tool data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(jsonData))
				So(tool.Name, ShouldEqual, "updated_name")
				So(tool.Description, ShouldEqual, "Updated description")
				So(len(tool.Parameters), ShouldEqual, 1)
				So(tool.Strict, ShouldBeFalse)
			})

			Convey("And reading should return the updated data", func() {
				buffer := make([]byte, 1024)
				n, err := tool.Read(buffer)

				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)

				var parsed ToolData
				err = json.Unmarshal(buffer[:n], &parsed)
				So(err, ShouldBeNil)
				So(parsed.Name, ShouldEqual, "updated_name")
				So(parsed.Description, ShouldEqual, "Updated description")
			})
		})

		Convey("When writing invalid JSON", func() {
			invalidJSON := []byte(`{"name": "invalid" - broken json`)
			n, err := tool.Write(invalidJSON)

			Convey("Then it should retain bytes but not update fields", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(invalidJSON))
				So(tool.Name, ShouldEqual, "original_name")
				So(tool.Description, ShouldEqual, "Original description")
			})
		})
	})
}

// TestToolClose tests the Close method of Tool
func TestToolClose(t *testing.T) {
	Convey("Given a Tool without a function", t, func() {
		tool := NewTool("test_tool", "Test description", []Parameter{})

		Convey("When closing the tool", func() {
			err := tool.Close()

			Convey("Then it should not error", func() {
				So(err, ShouldBeNil)
			})
		})
	})

	Convey("Given a Tool with a function", t, func() {
		tool := NewTool("test_tool", "Test description", []Parameter{})
		mockFn := NewMockFunction("test data")
		tool.WithFunction(mockFn)

		Convey("When closing the tool", func() {
			err := tool.Close()

			Convey("Then it should close the function", func() {
				So(err, ShouldBeNil)
				So(mockFn.WasClosed(), ShouldBeTrue)
			})
		})
	})
}

// TestToolWithFunction tests the WithFunction method
func TestToolWithFunction(t *testing.T) {
	Convey("Given a Tool", t, func() {
		tool := NewTool("test_tool", "Test description", []Parameter{})

		Convey("When adding a function", func() {
			mockFn := NewMockFunction("test data")
			result := tool.WithFunction(mockFn)

			Convey("Then the function should be set", func() {
				So(result, ShouldEqual, tool) // Should return self
				So(tool.fn, ShouldEqual, mockFn)
			})

			Convey("And the Function method should return it", func() {
				fn := tool.Function()
				So(fn, ShouldEqual, mockFn)
			})
		})
	})
}
