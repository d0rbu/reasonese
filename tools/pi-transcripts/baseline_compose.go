// Run with go run <this-file> from the pinned Compose checkout. No Docker daemon.
package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"

	"github.com/docker/compose/v5/pkg/watch"
)

func main() {
	root, err := os.MkdirTemp("", "compose-baseline-")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(root)
	matcher, err := watch.DockerIgnoreTesterFromContents(root, "*\n!keep.txt\n")
	if err != nil {
		panic(err)
	}
	ignored, err := matcher.Matches(filepath.Join(root, "ordinary.txt"))
	if err != nil || ignored {
		panic("unexpected pinned wildcard baseline")
	}
	patterns := []string{"*", "!keep.txt"}
	original := slices.Clone(patterns)
	if _, err = watch.NewDockerPatternMatcher(root, patterns); err != nil {
		panic(err)
	}
	mutated := !slices.Equal(original, patterns)
	if !mutated {
		panic("expected the pinned constructor's slice mutation")
	}
	if err := json.NewEncoder(os.Stdout).Encode(map[string]any{
		"kind": "pinned_source_baseline", "ordinary_file_ignored": ignored, "caller_slice_mutated": mutated,
	}); err != nil {
		panic(err)
	}
}
