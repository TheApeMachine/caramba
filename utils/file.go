package utils

import (
	"os"

	"github.com/charmbracelet/log"
)

func Workdir() string {
	dir, err := os.Getwd()

	if err != nil {
		log.Error(err)
		return ""
	}

	return dir
}

func ReadFile(path string) string {
	content, err := os.ReadFile(path)

	if err != nil {
		log.Error(err)
		return ""
	}

	return string(content)
}
