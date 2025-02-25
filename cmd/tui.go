package cmd

import (
	"fmt"

	"github.com/charmbracelet/bubbles/list"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/theapemachine/caramba/pkg/agent/examples"
	"github.com/theapemachine/errnie"
)

var (
	titleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFFDF5")).
			Background(lipgloss.Color("#25A065")).
			Padding(0, 1)

	statusStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "#04B575", Dark: "#04B575"})
)

// exampleItem represents an example that can be selected
type exampleItem struct {
	title       string
	description string
}

func (i exampleItem) Title() string       { return i.title }
func (i exampleItem) Description() string { return i.description }
func (i exampleItem) FilterValue() string { return i.title }

// exampleListModel is the model for our TUI
type exampleListModel struct {
	list          list.Model
	apiKey        string
	topic         string
	message       string
	task          string
	maxIterations int
	timeout       int
	choice        string
	quitting      bool
}

func (m exampleListModel) Init() tea.Cmd {
	return nil
}

func (m exampleListModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.list.SetWidth(msg.Width)
		m.list.SetHeight(msg.Height - 4) // Leaving room for status line
		return m, nil

	case tea.KeyMsg:
		switch keypress := msg.String(); keypress {
		case "ctrl+c", "q":
			m.quitting = true
			return m, tea.Quit

		case "enter":
			if i, ok := m.list.SelectedItem().(exampleItem); ok {
				m.choice = i.title

				// Run the selected example
				var err error
				switch i.title {
				case "research":
					err = examples.ResearchExample(m.apiKey, m.topic)
				case "chat":
					err = examples.ChatExample(m.apiKey, m.message)
				case "iteration":
					err = examples.IterationExample(m.apiKey, m.task, m.maxIterations, m.timeout)
				case "communication":
					err = examples.CommunicationExample(m.apiKey)
				case "memory":
					err = examples.MemoryExample(m.apiKey)
				}

				if err != nil {
					errnie.Error(err)
				}

				return m, tea.Quit
			}
		}
	}

	var cmd tea.Cmd
	m.list, cmd = m.list.Update(msg)
	return m, cmd
}

func (m exampleListModel) View() string {
	if m.quitting {
		return ""
	}

	return fmt.Sprintf(
		"\n%s\n\n%s\n\n%s",
		titleStyle.Render(" Caramba Examples "),
		m.list.View(),
		"Press q to quit, enter to run selected example",
	)
}

// runExampleTUI runs the TUI interface for selecting an example
func runExampleTUI(apiKey, topic, message, task string, maxIterations, timeout int) error {
	// Define the available examples
	items := []list.Item{
		exampleItem{
			title:       "research",
			description: "Research assistant example that can search and synthesize information",
		},
		exampleItem{
			title:       "chat",
			description: "Simple chat example with an AI assistant",
		},
		exampleItem{
			title:       "iteration",
			description: "Example demonstrating iterative improvement of a task",
		},
		exampleItem{
			title:       "communication",
			description: "Example showing communication between multiple agents",
		},
		exampleItem{
			title:       "memory",
			description: "Example demonstrating memory capabilities",
		},
	}

	// Create the list
	l := list.New(items, list.NewDefaultDelegate(), 0, 0)
	l.Title = "Select an example to run"
	l.SetShowStatusBar(false)
	l.SetFilteringEnabled(true)
	l.Styles.Title = titleStyle

	// Create the model
	m := exampleListModel{
		list:          l,
		apiKey:        apiKey,
		topic:         topic,
		message:       message,
		task:          task,
		maxIterations: maxIterations,
		timeout:       timeout,
	}

	// Run the program
	p := tea.NewProgram(m, tea.WithAltScreen())
	_, err := p.Run()
	return err
}
