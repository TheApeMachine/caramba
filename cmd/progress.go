package cmd

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	bubbleprogress "charm.land/bubbles/v2/progress"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/spf13/cobra"

	"github.com/theapemachine/caramba/pkg/qpool"
)

const (
	progressPadding  = 2
	progressMaxWidth = 80
	progressBuffer   = 256
	progressJobTTL   = time.Minute
	progressJobLimit = 24 * time.Hour
)

var progressDetailStyle = lipgloss.NewStyle().
	Foreground(lipgloss.Color("#626262")).
	Render

type qpoolEventMsg struct {
	event qpool.Event
}

type qpoolProgressStopMsg struct{}

type qpoolProgressProgram struct {
	program       *tea.Program
	subscription  *qpool.Subscription
	pool          *qpool.Q
	runResult     chan *qpool.QValue
	forwardResult chan *qpool.QValue
	restoreLogs   func()
	enabled       bool
}

type qpoolProgressModel struct {
	progress bubbleprogress.Model
	status   string
	detail   string
	active   bool
}

type qpoolProgressState struct {
	status  string
	detail  string
	percent float64
	known   bool
}

func runWithQPoolProgress(command *cobra.Command, execute func() error) error {
	progressProgram := newQPoolProgressProgram(command.Context(), command.ErrOrStderr())

	if !progressProgram.enabled {
		return execute()
	}

	progressProgram.Start(command.Context())
	defer progressProgram.Stop()

	return execute()
}

func newQPoolProgressProgram(ctx context.Context, writer io.Writer) *qpoolProgressProgram {
	progressModel := qpoolProgressModel{
		progress: bubbleprogress.New(bubbleprogress.WithDefaultBlend()),
		status:   "waiting for qpool events",
	}

	return &qpoolProgressProgram{
		program: tea.NewProgram(
			progressModel,
			tea.WithContext(ctx),
			tea.WithInput(nil),
			tea.WithOutput(writer),
		),
		enabled: isTerminalWriter(writer),
	}
}

func (progressProgram *qpoolProgressProgram) Start(ctx context.Context) {
	if progressProgram == nil || !progressProgram.enabled {
		return
	}

	progressProgram.subscription = qpool.Subscribe(progressBuffer)
	progressProgram.restoreLogs = qpool.SuppressLogging()
	progressProgram.pool = qpool.NewQ(
		ctx,
		2,
		2,
		&qpool.Config{
			SchedulingTimeout:  progressJobLimit,
			JobChannelCapacity: 4,
			Scaler:             nil,
		},
	)

	progressProgram.runResult = progressProgram.pool.Schedule(
		"cmd.progress.run",
		func(context.Context) (any, error) {
			_, err := progressProgram.program.Run()

			return nil, err
		},
		qpool.WithExecTimeout(progressJobLimit),
		qpool.WithTTL(progressJobTTL),
	)
	progressProgram.forwardResult = progressProgram.pool.Schedule(
		"cmd.progress.forward",
		func(jobCtx context.Context) (any, error) {
			progressProgram.Forward(jobCtx)

			return nil, nil
		},
		qpool.WithExecTimeout(progressJobLimit),
		qpool.WithTTL(progressJobTTL),
	)
}

func (progressProgram *qpoolProgressProgram) Forward(ctx context.Context) {
	if progressProgram == nil || progressProgram.subscription == nil {
		return
	}

	for {
		select {
		case <-ctx.Done():
			return
		case event, channelOpen := <-progressProgram.subscription.Events():
			if !channelOpen {
				return
			}

			progressProgram.program.Send(qpoolEventMsg{event: event})
		}
	}
}

func (progressProgram *qpoolProgressProgram) Stop() {
	if progressProgram == nil || !progressProgram.enabled {
		return
	}

	if progressProgram.subscription != nil {
		progressProgram.subscription.Close()
	}

	progressProgram.program.Send(qpoolProgressStopMsg{})

	if !progressProgram.wait(progressProgram.runResult, time.Second) {
		progressProgram.program.Kill()
	}

	progressProgram.wait(progressProgram.forwardResult, time.Second)

	if progressProgram.pool != nil {
		progressProgram.pool.Close()
		progressProgram.pool = nil
	}

	if progressProgram.restoreLogs != nil {
		progressProgram.restoreLogs()
		progressProgram.restoreLogs = nil
	}
}

func (progressProgram *qpoolProgressProgram) wait(
	result <-chan *qpool.QValue,
	timeout time.Duration,
) bool {
	if result == nil {
		return true
	}

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case <-timer.C:
		return false
	case value, channelOpen := <-result:
		if !channelOpen || value == nil {
			return true
		}

		return value.Error == nil
	}
}

func (progressModel qpoolProgressModel) Init() tea.Cmd {
	return nil
}

func (progressModel qpoolProgressModel) Update(message tea.Msg) (tea.Model, tea.Cmd) {
	switch typedMessage := message.(type) {
	case tea.WindowSizeMsg:
		width := typedMessage.Width - progressPadding*2 - 4

		if width < 1 {
			width = 1
		}

		progressModel.progress.SetWidth(width)

		if progressModel.progress.Width() > progressMaxWidth {
			progressModel.progress.SetWidth(progressMaxWidth)
		}

		return progressModel, nil
	case qpoolEventMsg:
		state := qpoolProgressFromEvent(typedMessage.event, progressModel.progress.Percent())
		progressModel.status = state.status
		progressModel.detail = state.detail
		progressModel.active = true

		return progressModel, progressModel.progress.SetPercent(state.percent)
	case bubbleprogress.FrameMsg:
		var command tea.Cmd
		progressModel.progress, command = progressModel.progress.Update(typedMessage)

		return progressModel, command
	case qpoolProgressStopMsg:
		return progressModel, tea.Quit
	default:
		return progressModel, nil
	}
}

func (progressModel qpoolProgressModel) View() tea.View {
	if !progressModel.active {
		return tea.NewView("")
	}

	padding := strings.Repeat(" ", progressPadding)
	status := padding + progressModel.status

	if progressModel.detail != "" {
		status += " " + progressDetailStyle(progressModel.detail)
	}

	return tea.NewView("\n" +
		status + "\n" +
		padding + progressModel.progress.View() + "\n")
}

func qpoolProgressFromEvent(event qpool.Event, currentPercent float64) qpoolProgressState {
	if eventFieldBool(event, "done") {
		return qpoolProgressState{
			status:  event.Message,
			detail:  fieldDetail(event),
			percent: 1,
			known:   true,
		}
	}

	if current, total, ok := eventRatio(event, "read_bytes", "expected_bytes"); ok {
		return qpoolProgressState{
			status:  event.Message,
			detail:  fieldDetail(event),
			percent: eventProgressPercent(event, current, total),
			known:   true,
		}
	}

	if current, total, ok := eventRatio(event, "step", "steps"); ok {
		return qpoolProgressState{
			status:  event.Message,
			detail:  fieldDetail(event),
			percent: eventProgressPercent(event, current, total),
			known:   true,
		}
	}

	if current, total, ok := eventRatio(event, "token", "tokens"); ok {
		return qpoolProgressState{
			status:  event.Message,
			detail:  fieldDetail(event),
			percent: eventProgressPercent(event, current, total),
			known:   true,
		}
	}

	if current, total, ok := eventRatio(event, "file", "files"); ok {
		return qpoolProgressState{
			status:  event.Message,
			detail:  fieldDetail(event),
			percent: eventProgressPercent(event, current, total),
			known:   true,
		}
	}

	if current, total, ok := eventRatio(event, "term", "terms"); ok {
		return qpoolProgressState{
			status:  event.Message,
			detail:  fieldDetail(event),
			percent: eventProgressPercent(event, current, total),
			known:   true,
		}
	}

	return qpoolProgressState{
		status:  event.Message,
		detail:  fieldDetail(event),
		percent: nextUnknownProgress(currentPercent),
	}
}

func eventRatio(event qpool.Event, currentKey string, totalKey string) (float64, float64, bool) {
	current, currentOK := eventFieldFloat(event, currentKey)
	total, totalOK := eventFieldFloat(event, totalKey)

	return current, total, currentOK && totalOK && total > 0
}

func eventFieldFloat(event qpool.Event, key string) (float64, bool) {
	for _, field := range event.Fields {
		if field.Key != key {
			continue
		}

		return fieldValueFloat(field.Value)
	}

	return 0, false
}

func eventFieldBool(event qpool.Event, key string) bool {
	for _, field := range event.Fields {
		if field.Key != key {
			continue
		}

		typedValue, valid := field.Value.(bool)

		return valid && typedValue
	}

	return false
}

func fieldValueFloat(value any) (float64, bool) {
	switch typedValue := value.(type) {
	case int:
		return float64(typedValue), true
	case int8:
		return float64(typedValue), true
	case int16:
		return float64(typedValue), true
	case int32:
		return float64(typedValue), true
	case int64:
		return float64(typedValue), true
	case uint:
		return float64(typedValue), true
	case uint8:
		return float64(typedValue), true
	case uint16:
		return float64(typedValue), true
	case uint32:
		return float64(typedValue), true
	case uint64:
		return float64(typedValue), true
	case float32:
		return float64(typedValue), true
	case float64:
		return typedValue, true
	default:
		return 0, false
	}
}

func fieldDetail(event qpool.Event) string {
	for _, key := range []string{"path", "output", "model", "backend", "file"} {
		for _, field := range event.Fields {
			if field.Key != key {
				continue
			}

			return fmt.Sprint(field.Value)
		}
	}

	if event.Component == "" || event.Op == "" {
		return ""
	}

	return event.Component + "." + event.Op
}

func clampProgressPercent(current float64, total float64) float64 {
	percent := current / total

	if percent < 0 {
		return 0
	}

	if percent > 1 {
		return 1
	}

	return percent
}

func eventProgressPercent(event qpool.Event, current float64, total float64) float64 {
	if eventFieldBool(event, "done") {
		return 1
	}

	return clampProgressPercent(current, total)
}

func nextUnknownProgress(currentPercent float64) float64 {
	if currentPercent >= 0.92 {
		return currentPercent
	}

	return currentPercent + 0.03
}

func isTerminalWriter(writer io.Writer) bool {
	file, ok := writer.(*os.File)

	if !ok {
		return false
	}

	info, err := file.Stat()

	if err != nil {
		return false
	}

	return info.Mode()&os.ModeCharDevice != 0
}
