package tui

import (
	"strings"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
)

type ChatComponent struct {
	logger   *output.Logger
	hub      *hub.Queue
	viewport viewport.Model
	textarea textarea.Model
	messages []hub.Event
	focused  bool
	ready    bool
	width    int
	height   int
	stream   bool
	style    *Style
}

func NewChatComponent() *ChatComponent {
	ta := textarea.New()
	ta.Placeholder = "Type your message here..."
	ta.SetHeight(3)
	ta.SetWidth(60)
	ta.ShowLineNumbers = false
	vp := viewport.New(0, 0)

	return &ChatComponent{
		logger:   output.NewLogger(),
		hub:      hub.NewQueue(),
		viewport: vp,
		textarea: ta,
		messages: make([]hub.Event, 0),
		focused:  true,
		ready:    false,
		stream:   false,
		style:    NewStyle(),
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

	c.viewport.Width = contentWidth
	c.viewport.Height = viewportHeight
	c.textarea.SetWidth(contentWidth - 1)

	c.ready = true
}

func (c *ChatComponent) Focus(focus bool) tea.Cmd {
	c.focused = focus
	if focus {
		return c.textarea.Focus()
	}
	c.textarea.Blur()
	return nil
}

func (c *ChatComponent) IsFocused() bool {
	return c.focused
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

			c.textarea.Reset()
		}

		if c.focused {
			c.textarea, cmd = c.textarea.Update(msg)
			cmds = append(cmds, cmd)
		}
	case *hub.Event:
		c.messages = append(c.messages, *msg)

		var str strings.Builder
		for _, message := range c.messages {
			switch message.Type {
			case hub.EventTypeMessage:
				str.WriteString(
					strings.Join([]string{
						c.style.Label(LabelInfo, message.Origin),
						message.Message + "\n",
					}, " "),
				)
			case hub.EventTypeChunk:
				if !c.stream {
					c.stream = true
					str.WriteString(
						strings.Join([]string{
							c.style.Label(LabelInfo, message.Origin),
							message.Message,
						}, " "),
					)
				} else {
					str.WriteString(message.Message)
				}
			case hub.EventTypeStatus:
				str.WriteString(
					strings.Join([]string{
						c.style.Label(LabelStatus, message.Origin),
						message.Message + "\n",
					}, " "),
				)
			case hub.EventTypeToolCall:
				str.WriteString(
					strings.Join([]string{
						c.style.Label(LabelTool, message.Origin),
						message.Message + "\n",
					}, " "),
				)
			case hub.EventTypeError:
				str.WriteString(
					strings.Join([]string{
						c.style.Label(LabelError, message.Origin),
						message.Message + "\n",
					}, " "),
				)
			default:
				str.WriteString(
					strings.Join([]string{
						c.style.Label(LabelInfo, message.Origin),
						message.Message + "\n",
					}, " "),
				)
			}
		}

		c.viewport.SetContent(str.String())
		c.viewport.GotoBottom()

		if msg.Type == hub.EventTypeClear {
			c.messages = make([]hub.Event, 0)
			c.viewport.SetContent("")
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

func (c *ChatComponent) ScrollUp() {
	c.viewport.LineUp(3)
}

func (c *ChatComponent) ScrollDown() {
	c.viewport.LineDown(3)
}

func (c *ChatComponent) ClearMessages() tea.Cmd {
	return func() tea.Msg {
		return hub.NewEvent("system", "ui", "user", hub.EventTypeClear, "clear", map[string]string{})
	}
}
