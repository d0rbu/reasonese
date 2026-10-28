// Run with go run <this-file> from the pinned Prometheus checkout.
package main

import (
	"encoding/json"
	"os"

	"github.com/prometheus/common/model"
	"github.com/prometheus/prometheus/model/labels"
	"github.com/prometheus/prometheus/model/relabel"
)

func main() {
	input := labels.FromStrings("__name__", "http_requests_total", "zone", "edge")
	config := relabel.DefaultRelabelConfig
	config.SourceLabels = model.LabelNames{"optional_zone"}
	config.TargetLabel = "zone"
	config.NameValidationScheme = model.UTF8Validation
	if err := config.Validate(model.UTF8Validation); err != nil {
		panic(err)
	}
	builder := labels.NewBuilder(input)
	keep := relabel.ProcessBuilder(builder, &config)
	if !keep || builder.Labels().Has("zone") || input.Get("zone") != "edge" {
		panic("unexpected pinned relabel baseline")
	}
	if err := json.NewEncoder(os.Stdout).Encode(map[string]any{
		"kind": "pinned_source_baseline", "zone_present": false, "sample_kept": keep, "input_unchanged": true,
	}); err != nil {
		panic(err)
	}
}
