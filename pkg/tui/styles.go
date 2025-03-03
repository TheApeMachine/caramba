package tui

import (
	"github.com/charmbracelet/lipgloss"
)

var (
	red    = lipgloss.Color("#FA5252")
	green  = lipgloss.Color("#12B886")
	yellow = lipgloss.Color("#FAB005")
	blue   = lipgloss.Color("#4C6EF5")
	purple = lipgloss.Color("#6C50FF")
	white  = lipgloss.Color("#FFFFFF")
	grey   = lipgloss.Color("#868E96")
	gray   = lipgloss.Color("#495057")

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
)

type Style struct {
}

func NewStyle() *Style {
	return &Style{}
}

func (style *Style) Section(text string, width int) string {
	return lipgloss.NewStyle().
		Bold(true).
		Foreground(purple).
		BorderBottom(true).
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(gray).
		Height(1).
		Width(width).
		Render(text)
}

func (style *Style) Header(text string) string {
	return lipgloss.NewStyle().
		Bold(true).
		Foreground(purple).
		BorderBottom(true).
		MarginRight(1).
		Render(text)
}

func (style *Style) BrandLabel(text string) string {
	return lipgloss.NewStyle().
		Bold(true).
		Background(purple).
		Foreground(white).
		Padding(0, 1).
		Render(text)
}

func (style *Style) ErrorLabel(text string) string {
	return lipgloss.NewStyle().
		Bold(true).
		Background(red).
		Foreground(white).
		Padding(0, 1).
		Render(text)
}

func (style *Style) SuccessLabel(text string) string {
	return lipgloss.NewStyle().
		Bold(true).
		Background(green).
		Foreground(white).
		Padding(0, 1).
		Render(text)
}

func (style *Style) WarningLabel(text string) string {
	return lipgloss.NewStyle().
		Bold(true).
		Background(yellow).
		Foreground(white).
		Padding(0, 1).
		Render(text)
}

func (style *Style) InfoLabel(text string) string {
	return lipgloss.NewStyle().
		Bold(true).
		Background(blue).
		Foreground(white).
		Padding(0, 1).
		Render(text)
}

func (style *Style) ToolLabel(text string) string {
	return lipgloss.NewStyle().
		Bold(true).
		Background(grey).
		Foreground(white).
		Padding(0, 1).
		Render(text)
}

func (style *Style) SystemLabel(text string) string {
	return lipgloss.NewStyle().
		Bold(true).
		Background(gray).
		Foreground(white).
		Padding(0, 1).
		Render(text)
}
