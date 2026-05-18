//go:build linux

package tensor

import (
	"syscall"
)

/*
Linux mmap primitives. Anonymous private mappings with read/write
protections. MADV_DONTNEED reclaims physical pages without unmapping;
MADV_HUGEPAGE asks the kernel to back the mapping with transparent
huge pages.

Per spray-and-pray: this file compiles and the syscalls are real, but
the error paths are minimal. Phase 3 verification should add round-
trip tests on a Linux runner.
*/

func mmapAlloc(bytesNeeded int) []byte {
	buffer, err := syscall.Mmap(
		-1,
		0,
		bytesNeeded,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_ANON|syscall.MAP_PRIVATE,
	)

	if err != nil {
		// Fall back to heap allocation; the caller cannot easily
		// recover from a mmap failure on a fresh allocation.
		return make([]byte, bytesNeeded)
	}

	return buffer
}

func mmapFree(buffer []byte) {
	if len(buffer) == 0 {
		return
	}

	_ = syscall.Munmap(buffer)
}

func mmapAdviseDontNeed(buffer []byte) {
	if len(buffer) == 0 {
		return
	}

	// syscall.Madvise is available on Linux through golang.org/x/sys/unix.
	// Phase 3 follow-up: switch to unix.MadviseDontneed for a cleaner API.
	_ = madviseRaw(buffer, syscallMadvDontNeed)
}

func mmapAdviseHugePage(buffer []byte) {
	if len(buffer) == 0 {
		return
	}

	_ = madviseRaw(buffer, syscallMadvHugePage)
}

const (
	syscallMadvDontNeed = 4
	syscallMadvHugePage = 14
)

func madviseRaw(buffer []byte, advice int) error {
	_, _, errno := syscall.Syscall(
		syscall.SYS_MADVISE,
		uintptr(getBufferPointer(buffer)),
		uintptr(len(buffer)),
		uintptr(advice),
	)

	if errno != 0 {
		return errno
	}

	return nil
}
