package kubrick

import (
	"sync"
)

// DirtyRegion represents a rectangular area that needs updating
type DirtyRegion struct {
	StartRow, EndRow int
	StartCol, EndCol int
}

// Buffer represents a single screen buffer
type Buffer struct {
	data          [][]rune
	width         int
	height        int
	dirtyRegions  []DirtyRegion
	pooledRegions []DirtyRegion // Pre-allocated region pool
	regionMu      sync.RWMutex
}

// NewBuffer creates a new buffer with given dimensions
func NewBuffer(width, height int) *Buffer {
	b := &Buffer{
		width:         width,
		height:        height,
		pooledRegions: make([]DirtyRegion, 0, 16), // Pre-allocate space for dirty regions
	}
	b.Resize(width, height)
	return b
}

// MarkDirty marks a region as needing updates
func (b *Buffer) MarkDirty(region DirtyRegion) {
	b.regionMu.Lock()
	defer b.regionMu.Unlock()

	// Clamp region to buffer bounds
	if region.StartRow < 0 {
		region.StartRow = 0
	}
	if region.EndRow >= b.height {
		region.EndRow = b.height - 1
	}
	if region.StartCol < 0 {
		region.StartCol = 0
	}
	if region.EndCol >= b.width {
		region.EndCol = b.width - 1
	}

	// Try to merge with existing regions
	for i := 0; i < len(b.dirtyRegions); i++ {
		if b.regionsOverlap(region, b.dirtyRegions[i]) {
			// Merge regions
			b.dirtyRegions[i] = b.mergeRegions(region, b.dirtyRegions[i])
			return
		}
	}

	// No overlap found, add new region
	if len(b.pooledRegions) > 0 {
		// Reuse a pooled region
		b.dirtyRegions = append(b.dirtyRegions, region)
		b.pooledRegions = b.pooledRegions[:len(b.pooledRegions)-1]
	} else {
		b.dirtyRegions = append(b.dirtyRegions, region)
	}
}

// Clear removes all content and dirty regions
func (b *Buffer) Clear() {
	b.regionMu.Lock()
	defer b.regionMu.Unlock()

	// Return current buffer to pool and get a fresh one
	globalBufferPool.PutBuffer(b.data)
	b.data = globalBufferPool.GetBuffer(b.height, b.width)
	b.dirtyRegions = b.dirtyRegions[:0]
	b.MarkDirty(DirtyRegion{0, b.height - 1, 0, b.width - 1})
}

// GetDirtyRegions returns a copy of current dirty regions
func (b *Buffer) GetDirtyRegions() []DirtyRegion {
	b.regionMu.RLock()
	defer b.regionMu.RUnlock()

	result := make([]DirtyRegion, len(b.dirtyRegions))
	copy(result, b.dirtyRegions)
	return result
}

// ClearDirtyRegions removes all dirty region markers
func (b *Buffer) ClearDirtyRegions() {
	b.regionMu.Lock()
	b.pooledRegions = append(b.pooledRegions, b.dirtyRegions...)
	b.dirtyRegions = b.dirtyRegions[:0]
	b.regionMu.Unlock()
}

// regionsOverlap checks if two regions overlap or are adjacent
func (b *Buffer) regionsOverlap(r1, r2 DirtyRegion) bool {
	// Check if regions are within one row/column of each other
	rowOverlap := r1.StartRow <= r2.EndRow+1 && r2.StartRow <= r1.EndRow+1
	colOverlap := r1.StartCol <= r2.EndCol+1 && r2.StartCol <= r1.EndCol+1
	return rowOverlap && colOverlap
}

// mergeRegions combines two overlapping or adjacent regions
func (b *Buffer) mergeRegions(r1, r2 DirtyRegion) DirtyRegion {
	return DirtyRegion{
		StartRow: min(r1.StartRow, r2.StartRow),
		EndRow:   max(r1.EndRow, r2.EndRow),
		StartCol: min(r1.StartCol, r2.StartCol),
		EndCol:   max(r1.EndCol, r2.EndCol),
	}
}

// Write writes content to the buffer at specified position
func (b *Buffer) Write(row, col int, content []rune) {
	if row < 0 || row >= b.height || col < 0 || col >= b.width {
		return
	}

	endCol := min(col+len(content), b.width)
	copy(b.data[row][col:endCol], content[:endCol-col])

	b.MarkDirty(DirtyRegion{
		StartRow: row,
		EndRow:   row,
		StartCol: col,
		EndCol:   endCol - 1,
	})
}

// WriteString writes a string to the buffer at specified position
func (b *Buffer) WriteString(row, col int, content string) {
	b.Write(row, col, []rune(content))
}

// CopyFrom copies content from another buffer
func (b *Buffer) CopyFrom(other *Buffer) {
	minHeight := min(b.height, other.height)
	minWidth := min(b.width, other.width)

	for i := 0; i < minHeight; i++ {
		copy(b.data[i][:minWidth], other.data[i][:minWidth])
	}
}

// Resize adjusts buffer size, preserving content where possible
func (b *Buffer) Resize(width, height int) {
	if width == b.width && height == b.height {
		return
	}

	// Get new buffer from pool
	newData := globalBufferPool.GetBuffer(height, width)

	// Copy existing content
	minHeight := min(b.height, height)
	minWidth := min(b.width, width)
	for i := 0; i < minHeight; i++ {
		copy(newData[i][:minWidth], b.data[i][:minWidth])
	}

	// Return old buffer to pool
	globalBufferPool.PutBuffer(b.data)

	b.data = newData
	b.width = width
	b.height = height

	// Mark entire buffer as dirty after resize
	b.dirtyRegions = b.dirtyRegions[:0]
	b.MarkDirty(DirtyRegion{0, height - 1, 0, width - 1})
}

// Close releases the buffer's resources back to the pool
func (b *Buffer) Close() {
	if b.data != nil {
		globalBufferPool.PutBuffer(b.data)
		b.data = nil
	}
}

// CompareWith efficiently compares this buffer with another using SIMD
func (b *Buffer) CompareWith(other *Buffer) []DirtyRegion {
	if b.width != other.width || b.height != other.height {
		// If dimensions differ, mark entire buffer as dirty
		return []DirtyRegion{{0, b.height - 1, 0, b.width - 1}}
	}

	var regions []DirtyRegion
	var currentRegion *DirtyRegion

	// Compare each row
	for row := 0; row < b.height; row++ {
		// Use SIMD to find first difference in the row
		if !CompareBuffers(b.data[row], other.data[row]) {
			// Find exact differences using SIMD
			diffs := FindDifferences(b.data[row], other.data[row])

			for _, diff := range diffs {
				if currentRegion == nil {
					// Start new region
					regions = append(regions, DirtyRegion{
						StartRow: row,
						EndRow:   row,
						StartCol: diff.StartIndex,
						EndCol:   diff.StartIndex + diff.Length - 1,
					})
					currentRegion = &regions[len(regions)-1]
				} else if currentRegion.EndRow == row-1 &&
					currentRegion.StartCol == diff.StartIndex &&
					currentRegion.EndCol == diff.StartIndex+diff.Length-1 {
					// Extend current region
					currentRegion.EndRow = row
				} else {
					// Start new region
					regions = append(regions, DirtyRegion{
						StartRow: row,
						EndRow:   row,
						StartCol: diff.StartIndex,
						EndCol:   diff.StartIndex + diff.Length - 1,
					})
					currentRegion = &regions[len(regions)-1]
				}
			}
		} else {
			// Row matches exactly, end current region if any
			currentRegion = nil
		}
	}

	return regions
}
