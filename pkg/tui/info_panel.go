package tui

import (
	"fmt"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/theapemachine/caramba/pkg/hub"
)

// InfoPanelComponent handles the agent info display
type InfoPanelComponent struct {
	agentName      string
	modelName      string
	memoryEntries  string
	toolNames      []string
	lastExecuted   time.Time
	executionCount int
	width          int
	height         int
	ready          bool
	focused        bool
	style          *Style
}

// NewInfoPanelComponent creates a new info panel component
func NewInfoPanelComponent() *InfoPanelComponent {
	return &InfoPanelComponent{
		agentName:      "No agent connected",
		modelName:      "N/A",
		memoryEntries:  "0",
		toolNames:      []string{},
		lastExecuted:   time.Time{},
		executionCount: 0,
		ready:          false,
		focused:        false,
		style:          NewStyle(),
	}
}

// Init initializes the info panel component
func (i *InfoPanelComponent) Init() tea.Cmd {
	return nil
}

// SetSize updates the component dimensions
func (i *InfoPanelComponent) SetSize(width, height int) {
	i.width = width
	i.height = height
	i.ready = true
}

func (i *InfoPanelComponent) Width() int {
	return i.width
}

func (i *InfoPanelComponent) Height() int {
	return i.height
}

// Focus focuses or blurs the component
func (i *InfoPanelComponent) Focus(focus bool) tea.Cmd {
	i.focused = focus
	return nil
}

// IsFocused returns whether the component is focused
func (i *InfoPanelComponent) IsFocused() bool {
	return i.focused
}

// Update handles the component updates
func (i *InfoPanelComponent) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		// Calculate appropriate dimensions for this component
		// Use 30% of the window width
		infoPanelWidth := int(float64(msg.Width) * 0.3)
		// Use full height minus 2 for status bar
		infoPanelHeight := msg.Height - 2

		i.width = infoPanelWidth
		i.height = infoPanelHeight
		i.ready = true
	case *hub.Event:
		switch msg.Type {
		case hub.EventTypeStatus:
			i.agentName = msg.Origin

			if model, ok := msg.Meta["model"]; ok {
				i.modelName = model
			}
		case hub.EventTypeMetric:
			if msg.Role == "memory" {
				i.memoryEntries = msg.Message
			}
		}
	}

	return i, cmd
}

// View renders the info panel
func (i *InfoPanelComponent) View() string {
	if !i.ready {
		return "Initializing info panel..."
	}

	sections := []string{}

	// Agent Info Section with header
	agentInfoHeader := i.style.Header("AGENT INFO", i.width-4)
	agentInfoContent := lipgloss.JoinVertical(
		lipgloss.Left,
		AgentStyle.Render("Name")+"   "+i.agentName,
		AgentStyle.Render("Model")+"  "+i.modelName,
		AgentStyle.Render("Memory")+" "+i.memoryEntries,
	)
	sections = append(sections, agentInfoHeader, agentInfoContent)

	// Stats Section with header
	statsHeader := i.style.Header("STATS", i.width-4)

	var lastExecuted string
	if i.lastExecuted.IsZero() {
		lastExecuted = "Never"
	} else {
		lastExecuted = i.lastExecuted.Format("15:04:05")
	}

	statsContent := lipgloss.JoinVertical(
		lipgloss.Left,
		AgentStyle.Render("Execution count")+" "+fmt.Sprintf("%d", i.executionCount),
		AgentStyle.Render("Last executed")+"   "+lastExecuted,
	)
	sections = append(sections, statsHeader, statsContent)

	// Tools Section with header
	toolsHeader := i.style.Header("TOOLS", i.width-4)

	var toolsContent string
	if len(i.toolNames) == 0 {
		toolsContent = "No tools available"
	} else {
		toolsLines := []string{
			fmt.Sprintf("Available: %d", len(i.toolNames)),
		}

		// Add tool names with bullets
		for _, tool := range i.toolNames {
			toolsLines = append(toolsLines, "• "+tool)
		}

		toolsContent = lipgloss.JoinVertical(
			lipgloss.Left,
			toolsLines...,
		)
	}
	sections = append(sections, toolsHeader, toolsContent)

	// Join all sections vertically with padding
	allContent := lipgloss.JoinVertical(
		lipgloss.Left,
		sections...,
	)

	return allContent
}
