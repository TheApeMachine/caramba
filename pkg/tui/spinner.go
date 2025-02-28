package tui

import (
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
)

type Spinner struct {
	component spinner.Model
}

func NewSpinner() *Spinner {
	return &Spinner{
		component: spinner.New(),
	}
}

func (model *Spinner) Init() tea.Cmd {
	model.component.Spinner = spinner.Dot
	return model.component.Tick
}

func (model *Spinner) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	model.component, cmd = model.component.Update(msg)
	return model, cmd
}

func (model *Spinner) View() string {
	return model.component.View()
}
