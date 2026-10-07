// Package plan turns a validated TrainingJob into a human-readable dry-run summary.
// Phase 1 is static analysis only: it does NOT probe hardware or estimate VRAM fit, time, power or cost.
package plan

import (
	"fmt"
	"strings"

	"tinyforge.dev/forgectl/internal/jobspec"
)

type Plan struct {
	Job             *jobspec.Job
	ExamplesPerStep int
	Warnings        []string
	Notes           []string
}

func Build(j *jobspec.Job) *Plan {
	p := &Plan{Job: j}
	h := j.Spec.Hyperparameters
	p.ExamplesPerStep = h.BatchSize * h.GradAccum

	if j.Spec.Method == "full" {
		p.Warnings = append(p.Warnings, "method=full is not implemented by the worker yet; the job would be rejected")
	}
	if j.Spec.Backend == "soup" {
		p.Notes = append(p.Notes, "backend=soup: layer streaming (BETA) runs as a subprocess; needs TINYFORGE_SOUP_BIN and ~4 GB free RAM for an 8B model; val_loss gates fail closed")
	}
	if j.Spec.Resources.Nodes > 1 {
		p.Warnings = append(p.Warnings, "nodes>1: worker v1 supports a single node only")
	}
	if j.Spec.Resources.GPUs > 1 {
		p.Warnings = append(p.Warnings, "gpus>1: worker v1 supports 0 or 1 GPU per node")
	}
	if j.Spec.Model.Revision == "" {
		p.Warnings = append(p.Warnings, "model.revision is not pinned; results are not reproducible and a cluster with requirePinnedModels would reject it")
	}
	if h.WarmupSteps >= h.MaxSteps {
		p.Warnings = append(p.Warnings, "warmupSteps >= maxSteps: the learning rate never reaches its peak")
	}
	for _, e := range j.Spec.Export {
		if strings.HasPrefix(e, "gguf") && !contains(j.Spec.Export, "merged") {
			p.Notes = append(p.Notes, "gguf export implies a merged model; the worker will merge first")
			break
		}
	}
	p.Notes = append(p.Notes, "phase 1: no hardware probe, so VRAM fit, ETA, power and cost are not estimated yet")
	return p
}

func Render(p *Plan) string {
	j, h := p.Job, p.Job.Spec.Hyperparameters
	var b strings.Builder
	fmt.Fprintf(&b, "TrainingJob %s/%s\n", j.Metadata.Namespace, j.Metadata.Name)
	fmt.Fprintf(&b, "  backend:     %s\n", j.Spec.Backend)
	fmt.Fprintf(&b, "  method:      %s\n", j.Spec.Method)
	fmt.Fprintf(&b, "  model:       %s\n", j.Spec.Model.Base)
	fmt.Fprintf(&b, "  data:        %s (format %s, maxLen %d)\n", j.Spec.Data.Source, j.Spec.Data.Format, j.Spec.Data.MaxLen)
	fmt.Fprintf(&b, "  steps:       %d, %d examples/step (batch %d x accum %d), lr %g\n",
		h.MaxSteps, p.ExamplesPerStep, h.BatchSize, h.GradAccum, h.LearningRate)
	fmt.Fprintf(&b, "  resources:   %d node(s) x %d GPU(s)\n", j.Spec.Resources.Nodes, j.Spec.Resources.GPUs)
	fmt.Fprintf(&b, "  checkpoint:  every %d steps, resume=%t\n", j.Spec.Checkpoint.EverySteps, j.Spec.Checkpoint.Resume)
	fmt.Fprintf(&b, "  export:      %s\n", strings.Join(j.Spec.Export, ", "))
	for _, g := range j.Spec.Gates {
		fmt.Fprintf(&b, "  gate:        %s %s %g\n", g.Metric, g.Op, g.Value)
	}
	for _, w := range p.Warnings {
		fmt.Fprintf(&b, "WARN: %s\n", w)
	}
	for _, n := range p.Notes {
		fmt.Fprintf(&b, "NOTE: %s\n", n)
	}
	return b.String()
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}
