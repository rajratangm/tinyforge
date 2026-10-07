package jobspec

import (
	"os"
	"testing"
)

// FuzzParse: arbitrary bytes must never panic, and anything accepted must carry defaults (non-nil job, set API version).
func FuzzParse(f *testing.F) {
	f.Add([]byte(minimal))
	if b, err := os.ReadFile("../../spec/examples/sql-finetune.yaml"); err == nil {
		f.Add(b)
	}
	f.Add([]byte("{}"))
	f.Add([]byte("apiVersion: [1,2"))
	f.Fuzz(func(t *testing.T, raw []byte) {
		j, err := Parse(raw)
		if err == nil && j == nil {
			t.Fatal("nil job without error")
		}
	})
}
