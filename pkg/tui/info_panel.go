package tui

import (
	"sort"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/theapemachine/caramba/pkg/hub"
)

type AgentInfo struct {
	Name     string
	Status   string
	Model    string
	Process  string
	Metadata map[string]string
	Children map[string]*AgentInfo
}

type StoreInfo struct {
	Name      string
	Status    string
	Questions []string
	Keywords  []string
	Cyphers   []string
	Relations []string
}

// InfoPanelComponent handles the agent info display
type InfoPanelComponent struct {
	agents    map[string]*AgentInfo
	stores    map[string]*StoreInfo
	toolcalls map[string]*hub.Event
	width     int
	height    int
	ready     bool
	focused   bool
	style     *Style
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
		agents:    make(map[string]*AgentInfo),
		stores:    make(map[string]*StoreInfo),
		toolcalls: make(map[string]*hub.Event),
		ready:     false,
		focused:   false,
		style:     style,
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
		i.width = msg.Width - 3
		i.height = msg.Height
		i.ready = true
	case *hub.Event:
		switch msg.Topic {
		case hub.TopicTypeAgent:
			switch string(msg.Type) {
			case "status":
				if parent, ok := msg.Meta["parent"]; ok && parent != "" {
					i.agents[parent].Children[msg.Origin] = &AgentInfo{
						Name:     msg.Origin,
						Status:   msg.Message,
						Model:    msg.Meta["model"],
						Process:  msg.Meta["process"],
						Metadata: msg.Meta,
					}
				} else {
					i.agents[msg.Origin] = &AgentInfo{
						Name:     msg.Origin,
						Status:   msg.Message,
						Model:    msg.Meta["model"],
						Process:  msg.Meta["process"],
						Metadata: msg.Meta,
						Children: make(map[string]*AgentInfo),
					}
				}
			case "toolcall":
				i.toolcalls[msg.Origin] = msg
			}
		case hub.TopicTypeStore:
			switch string(msg.Type) {
			case "status":
				i.stores[msg.Origin] = &StoreInfo{
					Name:      msg.Origin,
					Status:    msg.Message,
					Questions: []string{},
					Keywords:  []string{},
					Cyphers:   []string{},
					Relations: []string{},
				}
			case "cypher":
				i.stores[msg.Origin].Cyphers = append(i.stores[msg.Origin].Cyphers, msg.Message)
			case "relation":
				i.stores[msg.Origin].Relations = append(i.stores[msg.Origin].Relations, msg.Message)
			case "question":
				i.stores[msg.Origin].Questions = append(i.stores[msg.Origin].Questions, msg.Message)
			case "keyword":
				i.stores[msg.Origin].Keywords = append(i.stores[msg.Origin].Keywords, msg.Message)
			}
		}
	}

	return i, cmd
}

func (i *InfoPanelComponent) View() string {
	var agents strings.Builder
	var stores strings.Builder
	var toolcalls strings.Builder

	for _, agent := range i.agents {
		agents.WriteString(i.renderAgent(agent))
	}

	for _, event := range i.stores {
		i.renderStore(event)
	}

	for _, event := range i.toolcalls {
		toolcalls.WriteString(i.renderEvent(event))
	}

	return lipgloss.JoinVertical(
		lipgloss.Left,
		i.style.Section("AGENTS", i.width),
		agents.String(),
		i.style.Section("STORES", i.width),
		stores.String(),
		i.style.Section("TOOLCALLS", i.width),
		toolcalls.String(),
	)
}

func (i *InfoPanelComponent) renderAgent(agent *AgentInfo) string {
	var out strings.Builder

	out.WriteString(
		lipgloss.JoinVertical(
			lipgloss.Left,
			lipgloss.JoinHorizontal(
				lipgloss.Center,
				i.style.Header(agent.Name),
				i.getStatusLabel(agent.Status),
			),
			i.renderMetadata(agent.Metadata),
		),
	)

	for _, child := range agent.Children {
		out.WriteString(i.renderAgent(child))
	}

	return out.String()
}

func (i *InfoPanelComponent) renderStore(store *StoreInfo) string {
	var out strings.Builder

	out.WriteString(
		lipgloss.JoinVertical(
			lipgloss.Left,
			i.style.Header(store.Name),
		),
	)

	for _, question := range store.Questions {
		out.WriteString(question + "\n")
	}

	for _, keyword := range store.Keywords {
		out.WriteString(keyword + "\n")
	}

	for _, cypher := range store.Cyphers {
		out.WriteString(cypher + "\n")
	}

	for _, relation := range store.Relations {
		out.WriteString(relation + "\n")
	}

	return out.String()
}

func (i *InfoPanelComponent) renderEvent(event *hub.Event) string {
	var out strings.Builder

	out.WriteString(
		lipgloss.JoinVertical(
			lipgloss.Left,
			lipgloss.JoinHorizontal(
				lipgloss.Center,
				i.style.Header(event.Origin),
				i.getStatusLabel(event.Message),
			),
			i.renderMetadata(event.Meta),
		),
	)

	return out.String()
}

func (i *InfoPanelComponent) renderMetadata(metadata map[string]string) string {
	var out strings.Builder

	// Get all keys from the map
	keys := make([]string, 0, len(metadata))
	for key := range metadata {
		keys = append(keys, key)
	}

	// Sort keys alphabetically
	sort.Strings(keys)

	// Find the longest key for alignment
	longestKeyLength := 0
	for _, key := range keys {
		if len(key) > longestKeyLength {
			longestKeyLength = len(key)
		}
	}

	// Iterate through sorted keys with aligned values
	for _, key := range keys {
		padding := strings.Repeat(" ", longestKeyLength-len(key))
		out.WriteString(key + padding + ": " + metadata[key] + "\n")
	}

	return out.String()
}

func (i *InfoPanelComponent) getStatusLabel(message string) string {
	switch message {
	case "success", "running", "ready", "online":
		return i.style.SuccessLabel("Running")
	case "warning", "pending":
		return i.style.WarningLabel("Warning")
	case "error", "offline":
		return i.style.ErrorLabel("Error")
	default:
		return i.style.SystemLabel(message)
	}
}
