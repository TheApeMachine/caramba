package tui

import (
	"container/ring"
	"fmt"
	"strconv"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/theapemachine/caramba/pkg/hub"
)

// InfoPanelComponent handles the agent info display
type InfoPanelComponent struct {
	agentName      string
	modelName      string
	memories       []string
	relationships  []string
	toolcalls      *ring.Ring
	lastExecuted   time.Time
	executionCount int
	qdrantStore    string
	neo4jStore     string
	width          int
	height         int
	ready          bool
	focused        bool
	style          *Style
}

type Toolcall struct {
	Origin string
	Name   string
	Args   string
}

// NewInfoPanelComponent creates a new info panel component
func NewInfoPanelComponent() *InfoPanelComponent {
	style := NewStyle()

	return &InfoPanelComponent{
		agentName:      "No agent connected",
		modelName:      "N/A",
		memories:       []string{},
		relationships:  []string{},
		toolcalls:      ring.New(5),
		lastExecuted:   time.Time{},
		executionCount: 0,
		qdrantStore:    style.Label(LabelWarning, "Pending..."),
		neo4jStore:     style.Label(LabelWarning, "Pending..."),
		ready:          false,
		focused:        false,
		style:          style,
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
		case hub.EventTypeToolCall:
			// Put the tool name at the beginning of the list
			i.toolcalls.Value = Toolcall{
				Origin: msg.Origin,
				Name:   msg.Role,
				Args:   msg.Message,
			}
			i.toolcalls = i.toolcalls.Next()
		case hub.EventTypeStatus:
			i.agentName = msg.Origin

			if model, ok := msg.Meta["model"]; ok {
				i.modelName = model
			}

			if msg.Origin == "qdrant" {
				i.qdrantStore = msg.Message
			}

			if msg.Origin == "neo4j" {
				i.neo4jStore = msg.Message
			}

			if msg.Role == "iteration" {
				i.executionCount, _ = strconv.Atoi(msg.Message)
			}
		case hub.EventTypeMetric:
			if msg.Origin == "qdrant" {
				i.memories = append(i.memories, msg.Message)
			}

			if msg.Origin == "neo4j" {
				i.relationships = append(i.relationships, msg.Message)
			}
		}
	}

	return i, cmd
}

func (i *InfoPanelComponent) View() string {
	if !i.ready {
		return "Initializing info panel..."
	}

	sections := []string{}

	storesHeader := i.style.Header("STORES", i.width-4)
	storesContent := lipgloss.JoinVertical(
		lipgloss.Left,
		i.style.Label(LabelInfo, "QDRANT")+" "+i.qdrantStore,
		i.style.Label(LabelInfo, "Memories")+"\n"+strings.Join(i.memories, "\n"),
		i.style.Label(LabelInfo, "NEO4J")+" "+i.neo4jStore,
		i.style.Label(LabelInfo, "Relationships")+"\n"+strings.Join(i.relationships, "\n"),
	)

	sections = append(sections, storesHeader, storesContent)

	agentInfoHeader := i.style.Header("AGENT INFO", i.width-4)
	agentInfoContent := lipgloss.JoinVertical(
		lipgloss.Left,
		AgentStyle.Render("Name")+"   "+i.agentName,
		AgentStyle.Render("Model")+"  "+i.modelName,
	)

	sections = append(sections, agentInfoHeader, agentInfoContent)

	statsHeader := i.style.Header("STATS", i.width-4)

	statsContent := lipgloss.JoinVertical(
		lipgloss.Left,
		AgentStyle.Render("ITERATION")+" "+fmt.Sprintf("%d", i.executionCount),
	)

	sections = append(sections, statsHeader, statsContent)

	toolsHeader := i.style.Header("TOOLS", i.width-4)

	// Only proceed if the ring has elements and current value is not nil
	if i.toolcalls.Len() > 0 {
		toolsContent := ""
		toolnames := []string{}

		// Check if current value is not nil before type assertion
		if i.toolcalls.Value != nil {
			currentTool, ok := i.toolcalls.Value.(Toolcall)
			if ok {
				// Safely iterate through the ring buffer
				i.toolcalls.Do(func(item any) {
					if item != nil {
						if tool, ok := item.(Toolcall); ok {
							toolnames = append(toolnames, tool.Name)
						}
					}
				})

				toolsContent = lipgloss.JoinVertical(
					lipgloss.Left,
					currentTool.Origin+" "+currentTool.Name+"\n"+currentTool.Args,
					strings.Join(toolnames, "\n"),
				)

				sections = append(sections, toolsHeader, toolsContent)
			}
		} else {
			// Just show the header with empty content if we have no valid tool calls
			sections = append(sections, toolsHeader, "No tool calls recorded")
		}
	} else {
		sections = append(sections, toolsHeader, "No tool calls recorded")
	}

	allContent := lipgloss.JoinVertical(
		lipgloss.Left,
		sections...,
	)

	return allContent
}
