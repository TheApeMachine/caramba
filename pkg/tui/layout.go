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
		logger: output.NewLogger(),
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
	layout.logger.Log("tui", "Initializing layout")
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

		// Calculate component dimensions based on window size
		chatWidth := int(float64(msg.Width)*0.7) - 1 // 70% of window width minus separator
		infoPanelWidth := msg.Width - chatWidth - 1  // Remaining width minus separator

		// Send sized window messages to components
		chatMsg := tea.WindowSizeMsg{Width: chatWidth, Height: msg.Height - 2}
		infoPanelMsg := tea.WindowSizeMsg{Width: infoPanelWidth, Height: msg.Height - 2}
		statusBarMsg := tea.WindowSizeMsg{Width: msg.Width, Height: 1}

		// Update chat with its dimensions
		var cmd tea.Cmd
		layout.components["chat"], cmd = layout.components["chat"].Update(chatMsg)
		cmds = append(cmds, cmd)

		// Update info panel with its dimensions
		layout.components["infoPanel"], cmd = layout.components["infoPanel"].Update(infoPanelMsg)
		cmds = append(cmds, cmd)

		// Update status bar with its dimensions
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
	// Get chat and info panel
	chat := layout.components["chat"]
	infoPanel := layout.components["infoPanel"]
	statusBar := layout.components["statusBar"]

	// Calculate widths based on a 70/30 split
	chatWidth := int(float64(layout.width) * 0.7)
	infoPanelWidth := layout.width - chatWidth - 1 // leave room for separator

	// Render chat with borders
	chatView := lipgloss.NewStyle().
		Border(windowBorder).
		BorderForeground(subtleColor).
		Width(chatWidth - 2).      // account for borders
		Height(layout.height - 5). // leave room for status bar
		Render(chat.View())

	// Render info panel with borders
	infoPanelView := lipgloss.NewStyle().
		Border(windowBorder).
		BorderForeground(subtleColor).
		Width(infoPanelWidth - 2). // account for borders
		Height(layout.height - 5). // leave room for status bar
		Render(infoPanel.View())

	// Join horizontally
	mainContent := lipgloss.JoinHorizontal(
		lipgloss.Top,
		chatView,
		infoPanelView,
	)

	// Status bar with border
	statusBarView := lipgloss.NewStyle().
		BorderTop(true).
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(subtleColor).
		Width(layout.width).
		Render(statusBar.View())

	// Join vertically
	return lipgloss.JoinVertical(
		lipgloss.Left,
		mainContent,
		statusBarView,
	)
}
