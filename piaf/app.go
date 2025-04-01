package piaf

/*
App is the main application struct that holds everything together.
*/
type App struct {
	editor *Editor
}

/*
NewApp sets up the application with everything it needs to run.
*/
func NewApp() *App {
	return &App{editor: NewEditor()}
}

/*
Run starts the application.
*/
func (a *App) Run() {
	a.editor.Run()
}
