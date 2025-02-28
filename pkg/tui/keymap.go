package tui

import (
	"github.com/charmbracelet/bubbles/key"
)

// KeyMap defines the keybindings for the application
type KeyMap struct {
	Help       key.Binding
	Quit       key.Binding
	Send       key.Binding
	FocusUp    key.Binding
	FocusDown  key.Binding
	ClearChat  key.Binding
	ToggleInfo key.Binding
}

// ShortHelp returns keybindings to be shown in the mini help view.
func (k KeyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Help, k.Quit, k.Send}
}

// FullHelp returns keybindings for the expanded help view.
func (k KeyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{
		{k.Help, k.Quit, k.Send},
		{k.FocusUp, k.FocusDown, k.ClearChat, k.ToggleInfo},
	}
}

// DefaultKeyMap returns default keybindings
func DefaultKeyMap() KeyMap {
	return KeyMap{
		Help: key.NewBinding(
			key.WithKeys("?"),
			key.WithHelp("?", "toggle help"),
		),
		Quit: key.NewBinding(
			key.WithKeys("ctrl+c", "esc"),
			key.WithHelp("ctrl+c/esc", "quit"),
		),
		Send: key.NewBinding(
			key.WithKeys("ctrl+enter", "enter"),
			key.WithHelp("enter", "send message"),
		),
		FocusUp: key.NewBinding(
			key.WithKeys("tab"),
			key.WithHelp("tab", "focus up"),
		),
		FocusDown: key.NewBinding(
			key.WithKeys("shift+tab"),
			key.WithHelp("shift+tab", "focus down"),
		),
		ClearChat: key.NewBinding(
			key.WithKeys("ctrl+l"),
			key.WithHelp("ctrl+l", "clear chat"),
		),
		ToggleInfo: key.NewBinding(
			key.WithKeys("ctrl+i"),
			key.WithHelp("ctrl+i", "toggle info panel"),
		),
	}
}
