package tui

import (
	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
	"github.com/theapemachine/caramba/pkg/stream"
)

type ChatComponent struct {
	consumer   *stream.Consumer
	logger     *output.Logger
	hub        *hub.Queue
	viewport   textarea.Model
	textarea   textarea.Model
	messages   []hub.Event
	focused    bool
	ready      bool
	width      int
	height     int
	stream     bool
	generating bool
	style      *Style
}

func NewChatComponent() *ChatComponent {
	vp := textarea.New()
	vp.CharLimit = 1000000

	ta := textarea.New()
	ta.Placeholder = "Type your message here..."
	ta.SetHeight(3)
	ta.SetWidth(60)
	ta.ShowLineNumbers = false

	return &ChatComponent{
		consumer:   stream.NewConsumer(),
		logger:     output.NewLogger(),
		hub:        hub.NewQueue(),
		viewport:   vp,
		textarea:   ta,
		messages:   make([]hub.Event, 0),
		focused:    true,
		ready:      false,
		stream:     false,
		generating: false,
		style:      NewStyle(),
	}
}

func (c *ChatComponent) Init() tea.Cmd {
	return tea.Batch(
		c.textarea.Focus(),
	)
}

func (c *ChatComponent) SetSize(width, height int) {
	c.width = width
	c.height = height
	contentWidth := width
	inputHeight := 4
	viewportHeight := height - inputHeight - 3

	c.viewport.SetWidth(contentWidth)
	c.viewport.SetHeight(viewportHeight)
	c.textarea.SetWidth(contentWidth - 1)

	c.ready = true
}

// getLabelType returns the appropriate label type for the event type
func (c *ChatComponent) getLabelType(eventType hub.EventType) Label {
	switch eventType {
	case hub.EventTypeStatus:
		return LabelStatus
	case hub.EventTypeToolCall:
		return LabelTool
	case hub.EventTypeError:
		return LabelError
	default:
		return LabelInfo
	}
}

// insertMessage adds a message to the viewport with appropriate styling
func (c *ChatComponent) insertMessage(origin, message string, addNewline bool) {
	suffix := ""
	if addNewline {
		suffix = "\n"
	}

	// Process the message and strip any ANSI codes
	processedMessage := c.consumer.Feed(message)
	cleanMessage := StripANSIComprehensive(processedMessage)

	c.viewport.InsertString(
		"[" + origin + "] " + cleanMessage + suffix,
	)
}

func (c *ChatComponent) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		c.SetSize(msg.Width, msg.Height)
	case tea.KeyMsg:
		if msg.String() == "tab" {
			c.hub.Add(hub.NewEvent(
				"user",
				"researcher",
				"user",
				hub.EventTypeMessage,
				c.textarea.Value(),
				map[string]string{},
			))

			c.insertMessage("user", c.textarea.Value(), true)
			c.textarea.Reset()
		}

		if c.focused {
			c.textarea, cmd = c.textarea.Update(msg)
			cmds = append(cmds, cmd)
		}
	case *hub.Event:
		switch msg.Type {
		case hub.EventTypeChunk:
			cmds = append(cmds, c.viewport.Focus())
			c.textarea.Blur()
			c.focused = true

			c.viewport.CursorEnd()

			c.generating = true
			if !c.stream {
				c.stream = true
				c.insertMessage(msg.Origin, msg.Message, false)
			} else {
				// Use more aggressive ANSI stripping for chunks
				cleanMessage := StripANSIComprehensive(msg.Message)
				c.viewport.InsertString(cleanMessage)
			}
		case hub.EventTypeError:
			c.insertMessage(msg.Origin, msg.Message, true)
		default:
			if msg.Message == "done" {
				c.generating = false
				c.viewport.Blur()
				c.focused = false
				cmds = append(cmds, c.textarea.Focus())
			} else {
				// Handle standard messages
				c.generating = true
				c.insertMessage(msg.Origin, msg.Message, true)
			}
		}
	}

	c.viewport, cmd = c.viewport.Update(msg)
	cmds = append(cmds, cmd)

	return c, tea.Batch(cmds...)
}

func (c *ChatComponent) View() string {
	if !c.ready {
		return "Initializing chat..."
	}

	return lipgloss.JoinVertical(
		lipgloss.Left,
		c.viewport.View(),
		c.ViewInput(),
	)
}

func (c *ChatComponent) ViewInput() string {
	textareaView := lipgloss.NewStyle().
		BorderTop(true).
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(subtleColor).
		Render(c.textarea.View())

	return textareaView
}
