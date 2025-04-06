package spinner

import (
	"bufio"
	"container/ring"
	"strconv"

	"github.com/theapemachine/caramba/kubrick/components"
)

var (
	defaultFrames = []rune{
		'⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏',
	}

	successFrame = '✓'
	failureFrame = '✗'
)

type Spinner struct {
	components.BaseComponent
	frames *ring.Ring
	label  string
}

func NewSpinner(label string) *Spinner {
	frames := ring.New(len(defaultFrames))
	for _, frame := range defaultFrames {
		frames.Value = frame
		frames = frames.Next()
	}

	return &Spinner{
		frames: frames,
		label:  label,
	}
}

func (spinner *Spinner) Next() {
	spinner.frames = spinner.frames.Next()
	spinner.SetDirty(true)
}

func (spinner *Spinner) Success() {
	spinner.frames.Value = successFrame
	spinner.SetDirty(true)
}

func (spinner *Spinner) Failure() {
	spinner.frames.Value = failureFrame
	spinner.SetDirty(true)
}

func (spinner *Spinner) Render(writer *bufio.Writer) error {
	rect := spinner.GetRect()

	// Move cursor to component position
	writer.WriteString("\033[" + strconv.Itoa(rect.Pos.Row) + ";" + strconv.Itoa(rect.Pos.Col) + "H")

	// Write current frame and label to our buffer
	frame := []byte(string(spinner.frames.Value.(rune)))
	if len(spinner.label) > 0 && rect.Size.Width > 2 {
		frame = append(frame, ' ')
		frame = append(frame, spinner.label...)
	}

	// Update our internal buffer
	spinner.Write(frame)

	// Write to output
	writer.Write(frame)

	spinner.SetDirty(false)
	return nil
}
