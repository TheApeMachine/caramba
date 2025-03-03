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
	viewportHeight := height - inputHeight - 1

	c.viewport.SetWidth(contentWidth)
	c.viewport.SetHeight(viewportHeight)
	c.textarea.SetWidth(contentWidth - 1)

	c.ready = true
}

func (c *ChatComponent) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		c.SetSize(msg.Width, msg.Height)
	case tea.KeyMsg:
		if msg.String() == "tab" {
			c.hub.Add(&hub.Event{
				Topic:   hub.TopicTypeTask,
				Type:    hub.EventTypeUser,
				Origin:  "user",
				Message: c.textarea.Value(),
			})

			c.textarea.Reset()
		}

		if c.focused {
			c.textarea, cmd = c.textarea.Update(msg)
			cmds = append(cmds, cmd)
		}
	case *hub.Event:
		switch msg.Topic {
		case hub.TopicTypeMessage:
			switch msg.Type {
			case hub.EventTypePrompt:
				c.viewport.InsertString(msg.Message)
			case hub.EventTypeChunk:
				c.viewport.InsertString(msg.Message)
			case hub.EventTypeResponse:
				c.viewport.InsertString(msg.Message)
			}
		}
	}

	c.viewport, cmd = c.viewport.Update(msg)
	cmds = append(cmds, cmd)

	return c, tea.Batch(cmds...)
}

func (c *ChatComponent) View() string {
	return lipgloss.JoinVertical(
		lipgloss.Left,
		c.viewport.View(),
		lipgloss.NewStyle().
			BorderTop(true).
			BorderStyle(lipgloss.NormalBorder()).
			BorderForeground(gray).
			Render(c.textarea.View()),
	)
}
