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
	style  *Style
}

func NewStatusBar() *StatusBar {
	return &StatusBar{
		status: "Ready",
		help:   "? for help • enter to send • esc to quit",
		ready:  false,
		style:  NewStyle(),
	}
}

func (statusbar *StatusBar) Init() tea.Cmd {
	return nil
}

func (statusbar *StatusBar) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		statusbar.width = msg.Width
		statusbar.ready = true
	case hub.Event:
		switch msg.Type {
		case hub.EventTypeStatus:
			statusbar.status = msg.Message
		}
	}

	return statusbar, cmd
}

func (statusbar *StatusBar) View() string {
	return lipgloss.NewStyle().
		Width(statusbar.width).
		Align(lipgloss.Center).
		PaddingTop(0).
		PaddingBottom(0).
		Render(lipgloss.JoinHorizontal(
			lipgloss.Center,
			statusbar.style.BrandLabel(statusbar.status),
			" • ",
			statusbar.style.SystemLabel(statusbar.help),
		))
}
