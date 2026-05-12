package asset

import (
	"testing"
)

func TestWalk(t *testing.T) {
	schemas, err := Walk("template/operation")
	if err != nil {
		t.Fatalf("Walk operation failed: %v", err)
	}
	if len(schemas) == 0 {
		t.Fatal("Expected schemas, got 0")
	}

	blocks, err := Walk("template/block")
	if err != nil {
		t.Fatalf("Walk block failed: %v", err)
	}
	if len(blocks) == 0 {
		t.Fatal("Expected blocks, got 0")
	}
    
    models, err := Walk("template/model")
	if err != nil {
		t.Fatalf("Walk model failed: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("Expected models, got 0")
	}
}
