package tensor

import "sync"

/*
Tier 3 allocator: ≥ 1 GiB allocations via anonymous mmap with
huge-page advice. Freed buffers go on an indefinite free list and are
never returned to the OS — weight-load workloads recur and the
amortized cost of keeping the virtual mapping is less than the cost
of remapping every time.

No-zero contract: re-acquired buffers contain indeterminate data
unless the caller uses NewZeroed.
*/

type largeBlock struct {
	bytes uintptr
	data  []byte
	next  *largeBlock
}

type largePool struct {
	mu   sync.Mutex
	head *largeBlock
}

var defaultLarge = &largePool{}

/*
mmapLarge returns an mmap'd buffer of at least bytesNeeded size, drawn
from the free list or freshly allocated. The buffer is huge-page-
advised on Linux (MADV_HUGEPAGE) and superpage-flagged on Darwin
(VM_FLAGS_SUPERPAGE_SIZE_ANY).
*/
func mmapLarge(bytesNeeded int) []byte {
	rounded := roundToHugePage(bytesNeeded)

	defaultLarge.mu.Lock()
	previous := (*largeBlock)(nil)
	cursor := defaultLarge.head

	for cursor != nil {
		if int(cursor.bytes) >= bytesNeeded {
			if previous == nil {
				defaultLarge.head = cursor.next
			} else {
				previous.next = cursor.next
			}

			defaultLarge.mu.Unlock()

			return cursor.data[:bytesNeeded:cursor.bytes]
		}

		previous = cursor
		cursor = cursor.next
	}
	defaultLarge.mu.Unlock()

	allocated := mmapAlloc(rounded)
	mmapAdviseHugePage(allocated)

	return allocated[:bytesNeeded:rounded]
}

/*
mmapLargeRelease parks a buffer on the indefinite free list.
MADV_DONTNEED reclaims the physical pages while keeping the virtual
mapping alive.
*/
func mmapLargeRelease(buffer []byte) {
	mmapAdviseDontNeed(buffer)

	block := &largeBlock{bytes: uintptr(cap(buffer)), data: buffer}

	defaultLarge.mu.Lock()
	block.next = defaultLarge.head
	defaultLarge.head = block
	defaultLarge.mu.Unlock()
}

/*
roundToHugePage rounds a byte count up to the next 2 MiB boundary. On
Linux this matches THP's preferred granularity; on Darwin superpages
are 2 MiB or 1 GiB depending on the variant.
*/
func roundToHugePage(bytesNeeded int) int {
	const hugePage = 2 * 1024 * 1024

	return (bytesNeeded + hugePage - 1) &^ (hugePage - 1)
}
