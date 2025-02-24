package utils

import (
	"fmt"
	"time"

	"github.com/google/uuid"
)

// GenerateID creates a unique identifier with timestamp prefix
func GenerateID() string {
	timestamp := time.Now().UTC().Format("20060102150405")
	uid := uuid.New().String()[:8]
	return fmt.Sprintf("%s-%s", timestamp, uid)
}
