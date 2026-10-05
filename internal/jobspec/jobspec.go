// Package jobspec loads, validates and normalises TrainingJob specs (tinyforge.dev/v1alpha1).
package jobspec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"

	"github.com/santhosh-tekuri/jsonschema/v6"
	"gopkg.in/yaml.v3"

	"tinyforge.dev/forgectl/spec"
)

type Job struct {
	APIVersion string   `json:"apiVersion"`
	Kind       string   `json:"kind"`
	Metadata   Metadata `json:"metadata"`
	Spec       Spec     `json:"spec"`
}

type Metadata struct {
	Name      string            `json:"name"`
	Namespace string            `json:"namespace"`
	Labels    map[string]string `json:"labels,omitempty"`
}

type Spec struct {
	Method          string          `json:"method"`
	Model           Model           `json:"model"`
	Data            Data            `json:"data"`
	Hyperparameters Hyperparameters `json:"hyperparameters"`
	Resources       Resources       `json:"resources"`
	Checkpoint      Checkpoint      `json:"checkpoint"`
	Gates           []Gate          `json:"gates,omitempty"`
	Export          []string        `json:"export"`
}

type Model struct {
	Base            string `json:"base"`
	Revision        string `json:"revision,omitempty"`
	TrustRemoteCode bool   `json:"trustRemoteCode"`
}

type Data struct {
	Source     string `json:"source"`
	Format     string `json:"format"`
	Limit      int    `json:"limit,omitempty"`
	ValPercent int    `json:"valPercent"`
	MaxLen     int    `json:"maxLen"`
}

type Hyperparameters struct {
	MaxSteps          int     `json:"maxSteps"`
	LearningRate      float64 `json:"learningRate"`
	WarmupSteps       int     `json:"warmupSteps"`
	GradClip          float64 `json:"gradClip"`
	BatchSize         int     `json:"batchSize"`
	GradAccum         int     `json:"gradAccum"`
	LoraR             int     `json:"loraR"`
	LoraAlpha         int     `json:"loraAlpha"`
	LoraDropout       float64 `json:"loraDropout"`
	GradCheckpointing bool    `json:"gradCheckpointing"`
	Seed              int     `json:"seed"`
}

type Resources struct {
	GPUs      int     `json:"gpus"`
	Nodes     int     `json:"nodes"`
	GPUModel  string  `json:"gpuModel,omitempty"`
	MinVRAMGB float64 `json:"minVramGb,omitempty"`
	CPU       string  `json:"cpu,omitempty"`
	Memory    string  `json:"memory,omitempty"`
}

type Checkpoint struct {
	EverySteps int    `json:"everySteps"`
	Resume     bool   `json:"resume"`
	Output     string `json:"output,omitempty"`
}

type Gate struct {
	Metric string  `json:"metric"`
	Op     string  `json:"op"`
	Value  float64 `json:"value"`
}

// Load reads a YAML (or JSON) file, validates it against the embedded schema and returns the job with defaults applied.
func Load(path string) (*Job, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(raw)
}

func Parse(raw []byte) (*Job, error) {
	var doc any
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("parse: %w", err)
	}
	// Round-trip through JSON so the validator sees plain JSON types.
	asJSON, err := json.Marshal(doc)
	if err != nil {
		return nil, fmt.Errorf("parse: %w", err)
	}
	if err := validate(asJSON); err != nil {
		return nil, err
	}
	j := defaults()
	dec := json.NewDecoder(bytes.NewReader(asJSON))
	dec.DisallowUnknownFields()
	if err := dec.Decode(j); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return j, nil
}

func validate(asJSON []byte) error {
	schemaDoc, err := jsonschema.UnmarshalJSON(bytes.NewReader(spec.JobSpecSchema))
	if err != nil {
		return fmt.Errorf("embedded schema: %w", err)
	}
	c := jsonschema.NewCompiler()
	if err := c.AddResource("jobspec.json", schemaDoc); err != nil {
		return fmt.Errorf("embedded schema: %w", err)
	}
	sch, err := c.Compile("jobspec.json")
	if err != nil {
		return fmt.Errorf("embedded schema: %w", err)
	}
	inst, err := jsonschema.UnmarshalJSON(bytes.NewReader(asJSON))
	if err != nil {
		return err
	}
	if err := sch.Validate(inst); err != nil {
		return fmt.Errorf("invalid spec: %w", err)
	}
	return nil
}

// defaults mirrors the "default" values in the schema (kept in sync by TestDefaultsMatchSchema).
func defaults() *Job {
	return &Job{
		Metadata: Metadata{Namespace: "default"},
		Spec: Spec{
			Data: Data{Format: "text", ValPercent: 5, MaxLen: 512},
			Hyperparameters: Hyperparameters{
				MaxSteps: 300, LearningRate: 2e-4, WarmupSteps: 20, GradClip: 1.0,
				BatchSize: 4, GradAccum: 4, LoraR: 16, LoraAlpha: 32, Seed: 1337,
			},
			Resources:  Resources{GPUs: 1, Nodes: 1},
			Checkpoint: Checkpoint{EverySteps: 50, Resume: true},
			Export:     []string{"adapter"},
		},
	}
}
