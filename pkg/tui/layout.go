package tui

import (
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/theapemachine/caramba/pkg/output"
)

// Layout handles the overall UI layout
type Layout struct {
	logger     *output.Logger
	width      int
	height     int
	focused    string
	components map[string]tea.Model
	style      *Style
}

// NewLayout creates a new Layout instance
func NewLayout() *Layout {
	return &Layout{
		components: map[string]tea.Model{
			"chat":      tea.Model(NewChatComponent()),
			"infoPanel": tea.Model(NewInfoPanelComponent()),
			"statusBar": tea.Model(NewStatusBar()),
		},
		focused: "chat", // Focus the chat by default
		style:   NewStyle(),
	}
}

func (layout *Layout) Init() tea.Cmd {
	layout.components["chat"].Init()
	layout.components["infoPanel"].Init()
	layout.components["statusBar"].Init()
	return nil
}

func (layout *Layout) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		layout.width = msg.Width
		layout.height = msg.Height

		chatWidth := int(float64(msg.Width)*0.7) - 1
		infoPanelWidth := msg.Width - chatWidth - 1

		chatMsg := tea.WindowSizeMsg{Width: chatWidth, Height: msg.Height - 2}
		infoPanelMsg := tea.WindowSizeMsg{Width: infoPanelWidth, Height: msg.Height - 2}
		statusBarMsg := tea.WindowSizeMsg{Width: msg.Width, Height: 1}

		var cmd tea.Cmd
		layout.components["chat"], cmd = layout.components["chat"].Update(chatMsg)
		cmds = append(cmds, cmd)

		layout.components["infoPanel"], cmd = layout.components["infoPanel"].Update(infoPanelMsg)
		cmds = append(cmds, cmd)

		layout.components["statusBar"], cmd = layout.components["statusBar"].Update(statusBarMsg)
		cmds = append(cmds, cmd)

		return layout, tea.Batch(cmds...)
	}

	// Update all components by passing them the message
	for name, component := range layout.components {
		var cmd tea.Cmd
		layout.components[name], cmd = component.Update(msg)
		cmds = append(cmds, cmd)
	}

	return layout, tea.Batch(cmds...)
}

func (layout *Layout) View() string {
	chat := layout.components["chat"]
	infoPanel := layout.components["infoPanel"]
	statusBar := layout.components["statusBar"]

	chatWidth := int(float64(layout.width) * 0.7)
	infoPanelWidth := layout.width - chatWidth - 1

	return lipgloss.JoinVertical(
		lipgloss.Left,
		lipgloss.JoinHorizontal(
			lipgloss.Top,
			lipgloss.NewStyle().
				Border(windowBorder).
				BorderForeground(gray).
				Width(chatWidth-2).
				Height(layout.height-5).
				Render(chat.View()),
			lipgloss.NewStyle().
				Border(windowBorder).
				BorderForeground(gray).
				Width(infoPanelWidth-2).
				Height(layout.height-3).
				Render(infoPanel.View()),
		),
		lipgloss.NewStyle().
			Width(layout.width).
			Render(statusBar.View()),
	)
}
