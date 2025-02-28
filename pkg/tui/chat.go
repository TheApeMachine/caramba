package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
)

// ChatComponent handles the chat functionality
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

// NewChatComponent creates a new chat component
func NewChatComponent() *ChatComponent {
	// Initialize textarea
	ta := textarea.New()
	ta.Placeholder = "Type your message here..."
	ta.SetHeight(3)
	ta.SetWidth(60)
	ta.ShowLineNumbers = false

	// Initialize viewport
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

// Init initializes the chat component
func (c *ChatComponent) Init() tea.Cmd {
	return tea.Batch(
		c.textarea.Focus(),
	)
}

// SetSize updates the component dimensions
func (c *ChatComponent) SetSize(width, height int) {
	c.width = width
	c.height = height

	// Calculate content width accounting for the border
	contentWidth := width

	// Reserve space for the input with border
	inputHeight := 4 // 3 for textarea + 1 for border

	// Viewport takes remaining height
	viewportHeight := height - inputHeight - 3

	// Set viewport and textarea dimensions
	c.viewport.Width = contentWidth
	c.viewport.Height = viewportHeight
	c.textarea.SetWidth(contentWidth - 1)

	c.ready = true
}

// Focus focuses or blurs the component
func (c *ChatComponent) Focus(focus bool) tea.Cmd {
	c.focused = focus
	if focus {
		return c.textarea.Focus()
	}
	c.textarea.Blur()
	return nil
}

// IsFocused returns whether the component is focused
func (c *ChatComponent) IsFocused() bool {
	return c.focused
}

// Update handles the component updates
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
				"prompt",
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
		c.logger.Log(fmt.Sprintf("Received hub event: %s (%s)", msg.String(), msg.Type))

		// Add message to the list regardless of type
		c.messages = append(c.messages, *msg)

		// Format all messages
		var str strings.Builder
		for _, message := range c.messages {
			// Format based on message type
			switch message.Type {
			case hub.EventTypeMessage:
				// Format regular messages
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

// View renders the chat component
func (c *ChatComponent) View() string {
	if !c.ready {
		return "Initializing chat..."
	}

	// Just join the viewport view and input field directly
	return lipgloss.JoinVertical(
		lipgloss.Left,
		c.viewport.View(),
		c.ViewInput(),
	)
}

// ViewInput renders the input field
func (c *ChatComponent) ViewInput() string {
	// Add a border above the textarea to visually separate it from the chat messages
	textareaView := lipgloss.NewStyle().
		BorderTop(true).
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(subtleColor).
		Render(c.textarea.View())

	return textareaView
}

// scrollUp scrolls the viewport up
func (c *ChatComponent) ScrollUp() {
	c.viewport.LineUp(3)
}

// scrollDown scrolls the viewport down
func (c *ChatComponent) ScrollDown() {
	c.viewport.LineDown(3)
}

// ClearMessages returns a command to clear messages
func (c *ChatComponent) ClearMessages() tea.Cmd {
	return func() tea.Msg {
		return hub.NewEvent("system", "ui", "user", hub.EventTypeClear, "clear", map[string]string{})
	}
}
