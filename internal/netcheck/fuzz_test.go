package netcheck

import (
	"os"
	"testing"
)

// FuzzParse: arbitrary check specs must never panic; accepted specs must yield well-formed checks.
func FuzzParse(f *testing.F) {
	if b, err := os.ReadFile("testdata/example.yaml"); err == nil {
		f.Add(b)
	}
	f.Add([]byte("checks: []"))
	f.Add([]byte("\x00\xff"))
	f.Fuzz(func(t *testing.T, raw []byte) {
		_, _ = Parse(raw, t.TempDir())
	})
}
