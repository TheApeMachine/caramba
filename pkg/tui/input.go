package tui

import (
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
)

type InputComponent struct {
	textarea textarea.Model
	viewport viewport.Model
}

func NewInputComponent() *InputComponent {
	ta := textarea.New()

	// Create viewport with default dimensions (will be sized later)
	vp := viewport.New(0, 0)

	return &InputComponent{
		textarea: ta,
		viewport: vp,
	}
}

func (i *InputComponent) Init() tea.Cmd {
	return tea.Batch(
		i.viewport.Init(),
	)
}

func (i *InputComponent) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	return i, nil
}

func (i *InputComponent) View() string {
	return i.textarea.View()
}

func (i *InputComponent) Focus() tea.Cmd {
	return i.textarea.Focus()
}
