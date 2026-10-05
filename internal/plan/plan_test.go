package plan

import (
	"strings"
	"testing"

	"tinyforge.dev/forgectl/internal/jobspec"
)

const base = `
apiVersion: tinyforge.dev/v1alpha1
kind: TrainingJob
metadata: {name: t1}
spec:
  method: lora
  model: {base: org/model, revision: abc123}
  data: {source: org/data}
`

func build(t *testing.T, doc string) *Plan {
	t.Helper()
	j, err := jobspec.Parse([]byte(doc))
	if err != nil {
		t.Fatal(err)
	}
	return Build(j)
}

func hasWarning(p *Plan, sub string) bool {
	for _, w := range p.Warnings {
		if strings.Contains(w, sub) {
			return true
		}
	}
	return false
}

func TestCleanSpecHasNoWarnings(t *testing.T) {
	p := build(t, base)
	if len(p.Warnings) != 0 {
		t.Fatalf("unexpected warnings: %v", p.Warnings)
	}
	if p.ExamplesPerStep != 16 { // defaults: batch 4 x accum 4
		t.Fatalf("examples/step = %d, want 16", p.ExamplesPerStep)
	}
}

func TestWarnings(t *testing.T) {
	cases := map[string]struct{ doc, want string }{
		"unpinned model": {strings.Replace(base, ", revision: abc123", "", 1), "not pinned"},
		"multi gpu":      {base + "  resources: {gpus: 4}\n", "gpus>1"},
		"multi node":     {base + "  resources: {nodes: 2}\n", "nodes>1"},
		"full method":    {strings.Replace(base, "method: lora", "method: full", 1), "not implemented"},
		"warmup":         {base + "  hyperparameters: {maxSteps: 10, warmupSteps: 10}\n", "never reaches"},
	}
	for name, c := range cases {
		if p := build(t, c.doc); !hasWarning(p, c.want) {
			t.Errorf("%s: want a warning containing %q, got %v", name, c.want, p.Warnings)
		}
	}
}

func TestRenderMentionsGatesAndWarnings(t *testing.T) {
	out := Render(build(t, strings.Replace(base, ", revision: abc123", "", 1)+
		"  gates: [{metric: val_loss, op: '<', value: 1.5}]\n"))
	for _, s := range []string{"TrainingJob default/t1", "gate:        val_loss < 1.5", "WARN:", "NOTE:"} {
		if !strings.Contains(out, s) {
			t.Errorf("output missing %q:\n%s", s, out)
		}
	}
}
