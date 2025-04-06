package components

import (
	"bufio"
)

// StackLayout arranges components vertically or horizontally
type StackLayout struct {
	Vertical bool
	Spacing  int
}

func NewVerticalStackLayout(spacing int) *StackLayout {
	return &StackLayout{
		Vertical: true,
		Spacing:  spacing,
	}
}

func NewHorizontalStackLayout(spacing int) *StackLayout {
	return &StackLayout{
		Vertical: false,
		Spacing:  spacing,
	}
}

func (l *StackLayout) Arrange(container Container) {
	rect := container.GetRect()
	components := container.GetComponents()
	if len(components) == 0 {
		return
	}

	// Calculate total spacing needed
	totalSpacing := l.Spacing * (len(components) - 1)

	// Calculate available space for components
	var availableSpace int
	if l.Vertical {
		availableSpace = rect.Size.Height - totalSpacing
	} else {
		availableSpace = rect.Size.Width - totalSpacing
	}

	// Calculate size per component
	sizePerComponent := availableSpace / len(components)

	// Position components
	currentPos := rect.Pos
	for _, comp := range components {
		var compRect Rect
		if l.Vertical {
			compRect = Rect{
				Pos: currentPos,
				Size: Size{
					Width:  rect.Size.Width,
					Height: sizePerComponent,
				},
			}
			currentPos.Row += sizePerComponent + l.Spacing
		} else {
			compRect = Rect{
				Pos: currentPos,
				Size: Size{
					Width:  sizePerComponent,
					Height: rect.Size.Height,
				},
			}
			currentPos.Col += sizePerComponent + l.Spacing
		}
		comp.SetRect(compRect)
	}
}

// GridLayout arranges components in a grid pattern
type GridLayout struct {
	Rows    int
	Columns int
	Spacing int
}

func NewGridLayout(rows, columns, spacing int) *GridLayout {
	return &GridLayout{
		Rows:    rows,
		Columns: columns,
		Spacing: spacing,
	}
}

func (l *GridLayout) Arrange(container Container) {
	rect := container.GetRect()
	components := container.GetComponents()
	if len(components) == 0 {
		return
	}

	// Calculate cell dimensions
	cellWidth := (rect.Size.Width - (l.Spacing * (l.Columns - 1))) / l.Columns
	cellHeight := (rect.Size.Height - (l.Spacing * (l.Rows - 1))) / l.Rows

	// Position components
	for i, comp := range components {
		row := i / l.Columns
		col := i % l.Columns

		if row >= l.Rows {
			break // Don't position components that don't fit
		}

		compRect := Rect{
			Pos: Position{
				Row: rect.Pos.Row + (row * (cellHeight + l.Spacing)),
				Col: rect.Pos.Col + (col * (cellWidth + l.Spacing)),
			},
			Size: Size{
				Width:  cellWidth,
				Height: cellHeight,
			},
		}
		comp.SetRect(compRect)
	}
}

// BorderLayout arranges components in five regions: North, South, East, West, and Center
type BorderLayout struct {
	Spacing int
}

func NewBorderLayout(spacing int) *BorderLayout {
	return &BorderLayout{
		Spacing: spacing,
	}
}

const (
	BorderRegionNorth = iota
	BorderRegionSouth
	BorderRegionEast
	BorderRegionWest
	BorderRegionCenter
)

type BorderComponent struct {
	BaseComponent
	Component Component
	Region    int
}

func (bc *BorderComponent) Render(writer *bufio.Writer) error {
	return bc.Component.Render(writer)
}

func (l *BorderLayout) Arrange(container Container) {
	rect := container.GetRect()
	components := container.GetComponents()

	// Group components by region
	regions := make(map[int]Component)
	for _, comp := range components {
		if bc, ok := comp.(*BorderComponent); ok {
			regions[bc.Region] = bc.Component
		}
	}

	// Calculate dimensions for each region
	northHeight := 0
	southHeight := 0
	eastWidth := 0
	westWidth := 0

	if north := regions[BorderRegionNorth]; north != nil {
		northHeight = north.GetRect().Size.Height + l.Spacing
	}
	if south := regions[BorderRegionSouth]; south != nil {
		southHeight = south.GetRect().Size.Height + l.Spacing
	}
	if east := regions[BorderRegionEast]; east != nil {
		eastWidth = east.GetRect().Size.Width + l.Spacing
	}
	if west := regions[BorderRegionWest]; west != nil {
		westWidth = west.GetRect().Size.Width + l.Spacing
	}

	// Position components
	if north := regions[BorderRegionNorth]; north != nil {
		north.SetRect(Rect{
			Pos:  rect.Pos,
			Size: Size{Width: rect.Size.Width, Height: northHeight - l.Spacing},
		})
	}

	if south := regions[BorderRegionSouth]; south != nil {
		south.SetRect(Rect{
			Pos: Position{
				Row: rect.Pos.Row + rect.Size.Height - southHeight + l.Spacing,
				Col: rect.Pos.Col,
			},
			Size: Size{Width: rect.Size.Width, Height: southHeight - l.Spacing},
		})
	}

	if west := regions[BorderRegionWest]; west != nil {
		west.SetRect(Rect{
			Pos: Position{
				Row: rect.Pos.Row + northHeight,
				Col: rect.Pos.Col,
			},
			Size: Size{
				Width:  westWidth - l.Spacing,
				Height: rect.Size.Height - northHeight - southHeight,
			},
		})
	}

	if east := regions[BorderRegionEast]; east != nil {
		east.SetRect(Rect{
			Pos: Position{
				Row: rect.Pos.Row + northHeight,
				Col: rect.Pos.Col + rect.Size.Width - eastWidth + l.Spacing,
			},
			Size: Size{
				Width:  eastWidth - l.Spacing,
				Height: rect.Size.Height - northHeight - southHeight,
			},
		})
	}

	if center := regions[BorderRegionCenter]; center != nil {
		center.SetRect(Rect{
			Pos: Position{
				Row: rect.Pos.Row + northHeight,
				Col: rect.Pos.Col + westWidth,
			},
			Size: Size{
				Width:  rect.Size.Width - westWidth - eastWidth,
				Height: rect.Size.Height - northHeight - southHeight,
			},
		})
	}
}
