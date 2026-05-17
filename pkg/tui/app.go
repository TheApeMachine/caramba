package tui

import (
	tea "charm.land/bubbletea/v2"
	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
)

type App struct {
	form      *huh.Form
	styles    func(bool) *Styles
	hasDarkBg bool
	width     int
}

func NewApp() *App {
	return &App{
		form:      huh.NewForm(),
		styles:    NewStyles,
		hasDarkBg: false,
		width:     0,
	}
}

func (app *App) Init() tea.Cmd {
	return app.form.Init()
}

func (app *App) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	styles := app.styles(app.hasDarkBg)
	switch msg := msg.(type) {
	case tea.BackgroundColorMsg:
		app.hasDarkBg = msg.IsDark()
	case tea.WindowSizeMsg:
		app.width = min(msg.Width, maxWidth) - styles.Base.GetHorizontalFrameSize()
	case tea.KeyPressMsg:
		switch msg.String() {
		case "ctrl+c":
			return app, tea.Interrupt
		case "esc", "q":
			return app, tea.Quit
		}
	}

	var cmds []tea.Cmd

	// Process the form
	form, cmd := app.form.Update(msg)
	if f, ok := form.(*huh.Form); ok {
		app.form = f
		cmds = append(cmds, cmd)
	}

	if app.form.State == huh.StateCompleted {
		// Quit when the form is done.
		cmds = append(cmds, tea.Quit)
	}

	return app, tea.Batch(cmds...)
}

func (app *App) View() tea.View {
	s := app.styles(app.hasDarkBg)

	errors := app.form.Errors()
	header := app.appBoundaryView("Caramba")

	if len(errors) > 0 {
		header = app.appErrorBoundaryView(app.errorView())
	}

	form := ""
	status := ""

	body := lipgloss.JoinHorizontal(lipgloss.Left, form, status)
	footer := app.appBoundaryView(app.form.Help().ShortHelpView(app.form.KeyBinds()))

	if len(errors) > 0 {
		footer = app.appErrorBoundaryView("")
	}

	return tea.NewView(s.Base.Render(header + "\n" + body + "\n\n" + footer))
}

func (app *App) errorView() string {
	var s string
	for _, err := range app.form.Errors() {
		s += err.Error()
	}
	return s
}

func (app *App) appBoundaryView(text string) string {
	s := app.styles(app.hasDarkBg)
	return lipgloss.PlaceHorizontal(
		app.width,
		lipgloss.Left,
		s.HeaderText.Render(text),
		lipgloss.WithWhitespaceChars("/"),
		lipgloss.WithWhitespaceStyle(lipgloss.NewStyle().Foreground(s.Indigo)),
	)
}

func (app *App) appErrorBoundaryView(text string) string {
	s := app.styles(app.hasDarkBg)
	return lipgloss.PlaceHorizontal(
		app.width,
		lipgloss.Left,
		s.ErrorHeaderText.Render(text),
		lipgloss.WithWhitespaceChars("/"),
		lipgloss.WithWhitespaceStyle(lipgloss.NewStyle().Foreground(s.Red)),
	)
}
