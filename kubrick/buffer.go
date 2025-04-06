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
		data:          globalBufferPool.GetBuffer(height, width),
		pooledRegions: make([]DirtyRegion, 0, 16), // Pre-allocate space for dirty regions
	}
	return b
}

// MarkDirty marks a region as needing updates
func (buffer *Buffer) MarkDirty(region DirtyRegion) {
	buffer.regionMu.Lock()
	defer buffer.regionMu.Unlock()

	// Clamp region to buffer bounds
	if region.StartRow < 0 {
		region.StartRow = 0
	}
	if region.EndRow >= buffer.height {
		region.EndRow = buffer.height - 1
	}
	if region.StartCol < 0 {
		region.StartCol = 0
	}
	if region.EndCol >= buffer.width {
		region.EndCol = buffer.width - 1
	}

	// Try to merge with existing regions
	for i := 0; i < len(buffer.dirtyRegions); i++ {
		if buffer.regionsOverlap(region, buffer.dirtyRegions[i]) {
			// Merge regions
			buffer.dirtyRegions[i] = buffer.mergeRegions(region, buffer.dirtyRegions[i])
			return
		}
	}

	// No overlap found, add new region
	if len(buffer.pooledRegions) > 0 {
		// Reuse a pooled region
		buffer.dirtyRegions = append(buffer.dirtyRegions, region)
		buffer.pooledRegions = buffer.pooledRegions[:len(buffer.pooledRegions)-1]
	} else {
		buffer.dirtyRegions = append(buffer.dirtyRegions, region)
	}
}

// Clear removes all content and dirty regions
func (buffer *Buffer) Clear() {
	buffer.regionMu.Lock()
	defer buffer.regionMu.Unlock()

	// Return current buffer to pool and get a fresh one
	globalBufferPool.PutBuffer(buffer.data)
	buffer.data = globalBufferPool.GetBuffer(buffer.height, buffer.width)
	buffer.dirtyRegions = buffer.dirtyRegions[:0]
	buffer.MarkDirty(DirtyRegion{0, buffer.height - 1, 0, buffer.width - 1})
}

// GetDirtyRegions returns a copy of current dirty regions
func (buffer *Buffer) GetDirtyRegions() []DirtyRegion {
	buffer.regionMu.RLock()
	defer buffer.regionMu.RUnlock()

	result := make([]DirtyRegion, len(buffer.dirtyRegions))
	copy(result, buffer.dirtyRegions)
	return result
}

// ClearDirtyRegions removes all dirty region markers
func (buffer *Buffer) ClearDirtyRegions() {
	buffer.regionMu.Lock()
	buffer.pooledRegions = append(buffer.pooledRegions, buffer.dirtyRegions...)
	buffer.dirtyRegions = buffer.dirtyRegions[:0]
	buffer.regionMu.Unlock()
}

// regionsOverlap checks if two regions overlap or are adjacent
func (buffer *Buffer) regionsOverlap(r1, r2 DirtyRegion) bool {
	// Check if regions are within one row/column of each other
	rowOverlap := r1.StartRow <= r2.EndRow+1 && r2.StartRow <= r1.EndRow+1
	colOverlap := r1.StartCol <= r2.EndCol+1 && r2.StartCol <= r1.EndCol+1
	return rowOverlap && colOverlap
}

// mergeRegions combines two overlapping or adjacent regions
func (buffer *Buffer) mergeRegions(r1, r2 DirtyRegion) DirtyRegion {
	return DirtyRegion{
		StartRow: min(r1.StartRow, r2.StartRow),
		EndRow:   max(r1.EndRow, r2.EndRow),
		StartCol: min(r1.StartCol, r2.StartCol),
		EndCol:   max(r1.EndCol, r2.EndCol),
	}
}

// Write writes content to the buffer at specified position
func (buffer *Buffer) Write(row, col int, content []rune) {
	if row < 0 || row >= buffer.height || col < 0 || col >= buffer.width {
		return
	}

	endCol := min(col+len(content), buffer.width)
	copy(buffer.data[row][col:endCol], content[:endCol-col])

	buffer.MarkDirty(DirtyRegion{
		StartRow: row,
		EndRow:   row,
		StartCol: col,
		EndCol:   endCol - 1,
	})
}

// WriteString writes a string to the buffer at specified position
func (buffer *Buffer) WriteString(row, col int, content string) {
	buffer.Write(row, col, []rune(content))
}

// CopyFrom copies content from another buffer
func (buffer *Buffer) CopyFrom(other *Buffer) {
	minHeight := min(buffer.height, other.height)
	minWidth := min(buffer.width, other.width)

	for i := 0; i < minHeight; i++ {
		copy(buffer.data[i][:minWidth], other.data[i][:minWidth])
	}
}

// Resize adjusts buffer size, preserving content where possible
func (buffer *Buffer) Resize(width, height int) {
	if width == buffer.width && height == buffer.height {
		return
	}

	// Get new buffer from pool
	newData := globalBufferPool.GetBuffer(height, width)

	// Copy existing content
	minHeight := min(buffer.height, height)
	minWidth := min(buffer.width, width)
	for i := 0; i < minHeight; i++ {
		copy(newData[i][:minWidth], buffer.data[i][:minWidth])
	}

	// Return old buffer to pool
	globalBufferPool.PutBuffer(buffer.data)

	buffer.data = newData
	buffer.width = width
	buffer.height = height

	// Mark entire buffer as dirty after resize
	buffer.dirtyRegions = buffer.dirtyRegions[:0]
	buffer.MarkDirty(DirtyRegion{0, height - 1, 0, width - 1})
}

// Close releases the buffer's resources back to the pool
func (buffer *Buffer) Close() {
	if buffer.data != nil {
		globalBufferPool.PutBuffer(buffer.data)
		buffer.data = nil
	}
}

// CompareWith efficiently compares this buffer with another using SIMD
func (buffer *Buffer) CompareWith(other *Buffer) []DirtyRegion {
	if buffer.width != other.width || buffer.height != other.height {
		// If dimensions differ, mark entire buffer as dirty
		return []DirtyRegion{{0, buffer.height - 1, 0, buffer.width - 1}}
	}

	var regions []DirtyRegion
	var currentRegion *DirtyRegion

	// Compare each row
	for row := 0; row < buffer.height; row++ {
		// Use SIMD to find first difference in the row
		if !CompareBuffers(buffer.data[row], other.data[row]) {
			// Find exact differences using SIMD
			diffs := FindDifferences(buffer.data[row], other.data[row])

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
