//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "activation.h"
import "C"

import (
	"fmt"
	"unsafe"
)

func newRuntimePJRTConfig(platform string) (PJRTConfig, error) {
	pjrtConfig, err := NewPJRTConfig(platform)

	if err != nil {
		return PJRTConfig{}, err
	}

	if err := pjrtConfig.ValidateRuntime(); err != nil {
		return PJRTConfig{}, err
	}

	if err := configurePJRT(pjrtConfig); err != nil {
		return PJRTConfig{}, err
	}

	return pjrtConfig, nil
}

func configurePJRT(pjrtConfig PJRTConfig) error {
	pluginFile := pjrtConfig.ResolvedPluginFile()

	if pluginFile == "" {
		return fmt.Errorf("xla: resolved PJRT plugin path is required")
	}

	platformCString := C.CString(pjrtConfig.Platform)
	defer C.free(unsafe.Pointer(platformCString))

	pluginCString := C.CString(pluginFile)
	defer C.free(unsafe.Pointer(pluginCString))

	if rc := C.xla_configure_plugin(platformCString, pluginCString); rc != 0 {
		return fmt.Errorf("xla: configure PJRT plugin %q for %s failed", pluginFile, pjrtConfig.Platform)
	}

	return nil
}
