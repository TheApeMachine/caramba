package tui

import (
	"github.com/charmbracelet/lipgloss"
)

type Label string

const (
	LabelInfo    Label = "info"
	LabelSuccess Label = "success"
	LabelError   Label = "error"
	LabelWarning Label = "warning"
	LabelTool    Label = "tool"
	LabelStatus  Label = "status"
	LabelSystem  Label = "system"
)

var (
	// Colors
	subtleColor    = lipgloss.Color("#5C5C5C")
	highlightColor = lipgloss.Color("#7D56F4")
	specialColor   = lipgloss.Color("#4B7BEC")
	successColor   = lipgloss.Color("#2ECC71")
	errorColor     = lipgloss.Color("#E74C3C")
	warningColor   = lipgloss.Color("#F39C12")
	infoColor      = lipgloss.Color("#3498DB")
	accentColor    = lipgloss.Color("#FF6B81")

	windowBorder = lipgloss.Border{
		Top:         "─",
		Bottom:      "─",
		Left:        "│",
		Right:       "│",
		TopLeft:     "╭",
		TopRight:    "╮",
		BottomLeft:  "╰",
		BottomRight: "╯",
	}

	// Base styles
	TitleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(highlightColor).
			MarginLeft(1)

	WindowStyle = lipgloss.NewStyle().
			Border(windowBorder).
			BorderForeground(subtleColor).
			Padding(1, 2)

	MainStyle = lipgloss.NewStyle().
			Border(windowBorder).
			BorderForeground(subtleColor).
			MarginRight(1)

	AsideStyle = lipgloss.NewStyle().
			Border(windowBorder).
			BorderForeground(subtleColor).
			MarginLeft(1)

	FooterStyle = lipgloss.NewStyle().
			MarginTop(1)

	FocusedWindowStyle = lipgloss.NewStyle().
				Border(windowBorder).
				BorderForeground(highlightColor).
				Padding(1, 2)

	// Content styles
	UserStyle = lipgloss.NewStyle().
			Foreground(accentColor).
			Bold(true)

	AgentStyle = lipgloss.NewStyle().
			Foreground(successColor).
			Bold(true)

	SystemStyle = lipgloss.NewStyle().
			Foreground(infoColor).
			Italic(true)

	ErrorStyle = lipgloss.NewStyle().
			Foreground(errorColor).
			Bold(true)

	ToolStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#fd9644")).
			Italic(true)

	TimestampStyle = lipgloss.NewStyle().
			Foreground(subtleColor).
			Italic(true)

	// UI elements
	HeaderStyle = lipgloss.NewStyle().
			Foreground(highlightColor).
			Bold(true).
			BorderBottom(true).
			BorderStyle(lipgloss.NormalBorder()).
			BorderForeground(subtleColor).
			MarginTop(1)

	StatusBarStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "#343433", Dark: "#C1C6B2"}).
			Background(lipgloss.AdaptiveColor{Light: "#D9DCCF", Dark: "#353533"})

	// Main container style
	DocStyle = lipgloss.NewStyle().
			Padding(1, 2)

	// Help style
	HelpStyle = lipgloss.NewStyle().
			Foreground(subtleColor)

	// Button styles
	ButtonStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFFFFF")).
			Background(highlightColor).
			Padding(0, 3).
			MarginRight(1)

	DisabledButtonStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#FFFFFF")).
				Background(subtleColor).
				Padding(0, 3).
				MarginRight(1)
)

type Style struct {
}

func NewStyle() *Style {
	return &Style{}
}

func (style *Style) Header(text string, width int) string {
	return HeaderStyle.Width(width).Render(text)
}

func (style *Style) Main(text string, width int, height int) string {
	return MainStyle.
		Width(width).
		Height(height).
		MarginRight(1).
		Render(text)
}

func (style *Style) Aside(text string, width int, height int) string {
	return AsideStyle.
		Width(width).
		Height(height).
		MarginLeft(1).
		Render(text)
}

func (style *Style) Footer(text string, width int) string {
	return FooterStyle.Width(width).Render(text)
}

func (style *Style) StatusBar(text string) string {
	return StatusBarStyle.Render(text)
}

func (style *Style) Label(label Label, text string) string {
	switch label {
	case LabelInfo:
		return lipgloss.NewStyle().
			Foreground(infoColor).
			Render(text)
	case LabelSuccess:
		return lipgloss.NewStyle().
			Foreground(successColor).
			Render(text)
	case LabelError:
		return lipgloss.NewStyle().
			Foreground(errorColor).
			Render(text)
	case LabelWarning:
		return lipgloss.NewStyle().
			Foreground(warningColor).
			Render(text)
	case LabelTool:
		return lipgloss.NewStyle().
			Foreground(lipgloss.Color("#fd9644")).
			Render(text)
	case LabelStatus:
		return lipgloss.NewStyle().
			Foreground(highlightColor).
			Render(text)
	case LabelSystem:
		return lipgloss.NewStyle().
			Foreground(infoColor).
			Render(text)
	}

	return text
}
