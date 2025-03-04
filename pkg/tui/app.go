package tui

import (
	"context"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
)

// App is the main TUI application
type App struct {
	logger     *output.Logger
	hub        *hub.Queue
	keymap     KeyMap
	layout     tea.Model
	program    *tea.Program
	ready      bool
	loading    bool
	width      int
	height     int
	ctx        context.Context
	cancelFunc context.CancelFunc
}

// NewApp creates a new App instance
func NewApp() *App {
	keymap := DefaultKeyMap()
	ctx, cancelFunc := context.WithCancel(context.Background())

	app := &App{
		logger:     output.NewLogger(),
		hub:        hub.NewQueue(),
		keymap:     keymap,
		layout:     NewLayout(),
		ready:      false,
		loading:    false,
		ctx:        ctx,
		cancelFunc: cancelFunc,
	}

	go app.listenToHubEvents()
	return app
}

// Start starts the BubbleTea application
func (app *App) Start() error {
	// Create the program and store the reference
	p := tea.NewProgram(app, tea.WithAltScreen())
	app.program = p

	// Run the program
	_, err := p.Run()

	return err
}

// listenToHubEvents subscribes to hub events and forwards them to the UI
func (app *App) listenToHubEvents() {
	agentSubscription := app.hub.Subscribe(string(hub.TopicTypeAgent))
	messageSubscription := app.hub.Subscribe(string(hub.TopicTypeMessage))
	storeSubscription := app.hub.Subscribe(string(hub.TopicTypeStore))

	for {
		select {
		case <-app.ctx.Done():
			return
		case event := <-agentSubscription:
			if app.ready {
				app.program.Send(event)
			}
		case event := <-messageSubscription:
			if app.ready {
				app.program.Send(event)
			}
		case event := <-storeSubscription:
			if app.ready {
				app.program.Send(event)
			}
		}
	}
}

// Init initializes the BubbleTea application
func (app *App) Init() tea.Cmd {
	app.logger.Log("tui", "Initializing BubbleTea application")

	return tea.Batch(
		app.layout.Init(),
		tea.EnterAltScreen,
	)
}

// Update handles events and updates the application state
func (app *App) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		// Global key handlers
		switch {
		case key.Matches(msg, app.keymap.Quit):
			app.cancelFunc()
			return app, tea.Quit
		}

	case tea.WindowSizeMsg:
		app.height = msg.Height
		app.width = msg.Width

		if !app.ready {
			app.ready = true
		}
	}

	app.layout, cmd = app.layout.Update(msg)
	cmds = append(cmds, cmd)

	return app, tea.Batch(cmds...)
}

// View renders the application UI
func (app *App) View() string {
	if !app.ready {
		return "Starting Caramba TUI... Please resize your terminal if this message persists."
	}

	return app.layout.View()
}
