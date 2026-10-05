package jobspec

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"tinyforge.dev/forgectl/spec"
)

const minimal = `
apiVersion: tinyforge.dev/v1alpha1
kind: TrainingJob
metadata: {name: t1}
spec:
  method: lora
  model: {base: org/model}
  data: {source: org/data}
`

func TestMinimalSpecGetsDefaults(t *testing.T) {
	j, err := Parse([]byte(minimal))
	if err != nil {
		t.Fatal(err)
	}
	if j.Metadata.Namespace != "default" || j.Spec.Hyperparameters.MaxSteps != 300 ||
		j.Spec.Resources.GPUs != 1 || !j.Spec.Checkpoint.Resume || j.Spec.Data.MaxLen != 512 {
		t.Fatalf("defaults not applied: %+v", j)
	}
}

func TestInvalidSpecsRejected(t *testing.T) {
	cases := map[string]string{
		"bad method":         strings.Replace(minimal, "method: lora", "method: lora2", 1),
		"unknown field":      strings.Replace(minimal, "data: {source: org/data}", "data: {source: org/data, bogus: 1}", 1),
		"bad name":           strings.Replace(minimal, "name: t1", "name: Bad_Name", 1),
		"trustRemoteCode":    strings.Replace(minimal, "model: {base: org/model}", "model: {base: org/model, trustRemoteCode: true}", 1),
		"wrong apiVersion":   strings.Replace(minimal, "v1alpha1", "v9", 1),
		"missing data":       strings.Replace(minimal, "  data: {source: org/data}\n", "", 1),
		"zero steps":         minimal + "  hyperparameters: {maxSteps: 0}\n",
		"negative gpus":      minimal + "  resources: {gpus: -1}\n",
		"gate bad operator":  minimal + "  gates: [{metric: val_loss, op: '!=', value: 1}]\n",
		"duplicate export":   minimal + "  export: [adapter, adapter]\n",
		"not yaml":           "::: not yaml :::",
		"empty":              "",
	}
	for name, doc := range cases {
		if _, err := Parse([]byte(doc)); err == nil {
			t.Errorf("%s: expected an error, got none", name)
		}
	}
}

// The Go defaults are hand-copied from the schema; fail loudly if the two drift apart.
func TestDefaultsMatchSchema(t *testing.T) {
	var s struct {
		Properties struct {
			Spec struct {
				Properties map[string]struct {
					Properties map[string]struct {
						Default any `json:"default"`
					} `json:"properties"`
					Default any `json:"default"`
				} `json:"properties"`
			} `json:"spec"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(spec.JobSpecSchema, &s); err != nil {
		t.Fatal(err)
	}
	d := defaults().Spec
	got := map[string]map[string]any{
		"data": {"format": d.Data.Format, "valPercent": float64(d.Data.ValPercent), "maxLen": float64(d.Data.MaxLen)},
		"hyperparameters": {
			"maxSteps": float64(d.Hyperparameters.MaxSteps), "learningRate": d.Hyperparameters.LearningRate,
			"warmupSteps": float64(d.Hyperparameters.WarmupSteps), "gradClip": d.Hyperparameters.GradClip,
			"batchSize": float64(d.Hyperparameters.BatchSize), "gradAccum": float64(d.Hyperparameters.GradAccum),
			"loraR": float64(d.Hyperparameters.LoraR), "loraAlpha": float64(d.Hyperparameters.LoraAlpha),
			"loraDropout": d.Hyperparameters.LoraDropout, "gradCheckpointing": d.Hyperparameters.GradCheckpointing,
			"seed": float64(d.Hyperparameters.Seed),
		},
		"resources":  {"gpus": float64(d.Resources.GPUs), "nodes": float64(d.Resources.Nodes)},
		"checkpoint": {"everySteps": float64(d.Checkpoint.EverySteps), "resume": d.Checkpoint.Resume},
	}
	for section, fields := range got {
		for field, want := range fields {
			have := s.Properties.Spec.Properties[section].Properties[field].Default
			if !reflect.DeepEqual(have, want) {
				t.Errorf("%s.%s: schema default %v != Go default %v", section, field, have, want)
			}
		}
	}
}
