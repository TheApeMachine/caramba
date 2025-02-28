package tui

import (
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/theapemachine/caramba/pkg/hub"
)

type StatusBar struct {
	status string
	help   string
	width  int
	ready  bool
}

func NewStatusBar() *StatusBar {
	return &StatusBar{
		status: "Ready",
		help:   "? for help • enter to send • esc to quit",
		ready:  false,
	}
}

func (statusBar *StatusBar) Init() tea.Cmd {
	return nil
}

func (statusBar *StatusBar) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		statusBar.width = msg.Width
		statusBar.ready = true
	case hub.Event:
		switch msg.Type {
		case hub.EventTypeStatus:
			statusBar.status = msg.Message
		}
	}

	return statusBar, cmd
}

func (statusBar *StatusBar) View() string {
	if !statusBar.ready {
		return "Initializing status bar..."
	}

	// Create status section with minimal styling
	statusSection := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#64748b")).
		Bold(true).
		Render(statusBar.status)

	// Create help section with minimal styling
	helpSection := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#475569")).
		Render(statusBar.help)

	// Create a simple divider
	divider := " • "

	// Combine content
	content := lipgloss.JoinHorizontal(
		lipgloss.Center,
		statusSection,
		divider,
		helpSection,
	)

	// Create a thin border top
	barStyle := lipgloss.NewStyle().
		BorderTop(true).
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(subtleColor).
		Width(statusBar.width).
		Align(lipgloss.Center).
		PaddingTop(0).
		PaddingBottom(0)

	return barStyle.Render(content)
}
