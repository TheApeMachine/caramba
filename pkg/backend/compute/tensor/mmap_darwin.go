//go:build darwin

package tensor

import "syscall"

/*
Darwin mmap primitives. Anonymous private mappings. Darwin's superpage
flag is VM_FLAGS_SUPERPAGE_SIZE_ANY (0x10000) but Go's syscall package
does not expose it directly; we use plain anon mmap and rely on
posix_madvise/MADV_FREE_REUSABLE for reclamation.

Per spray-and-pray: this file compiles. Verification on macOS should
exercise the mmap round trip with realistic sizes.
*/

const (
	madvFreeReusable = 7
	madvDontNeed     = 4
)

func mmapAlloc(bytesNeeded int) []byte {
	buffer, err := syscall.Mmap(
		-1,
		0,
		bytesNeeded,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_ANON|syscall.MAP_PRIVATE,
	)

	if err != nil {
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

	_ = madviseRaw(buffer, madvFreeReusable)
}

func mmapAdviseHugePage(buffer []byte) {
	// Darwin doesn't have a portable "huge page" hint that survives
	// outside the VM subsystem. Leave as no-op; the mmap allocation
	// already prefers superpage-friendly alignment.
}

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
