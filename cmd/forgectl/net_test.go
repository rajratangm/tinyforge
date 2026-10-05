package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func closedLoopbackPort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()
	return port
}

func writeChecks(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "checks.yaml")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func net_(t *testing.T, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	var o, e bytes.Buffer
	code = runNetTo(args, &o, &e)
	return code, o.String(), e.String()
}

func TestNetCheckExitCodes(t *testing.T) {
	port := closedLoopbackPort(t)
	blocked := writeChecks(t, fmt.Sprintf("checks:\n  - {id: c, kind: egress-deny, target: \"127.0.0.1:%d\"}\n", port))
	failing := writeChecks(t, fmt.Sprintf("checks:\n  - {id: c, kind: tcp, target: \"127.0.0.1:%d\"}\n", port))
	warnOnly := writeChecks(t, fmt.Sprintf("checks:\n  - {id: c, kind: tcp, target: \"127.0.0.1:%d\", severity: warn}\n", port))
	cidr := writeChecks(t, "checks:\n  - {id: c, kind: tcp, target: \"10.0.0.0/24:443\"}\n")

	cases := []struct {
		name string
		args []string
		want int
		in   string // substring expected in stdout+stderr
	}{
		{"no arguments", nil, 2, "Usage"},
		{"unknown subcommand", []string{"scan"}, 2, "Usage"},
		{"neither -f nor --preset", []string{"check"}, 2, "exactly one"},
		{"both -f and --preset", []string{"check", "-f", blocked, "--preset", "local"}, 2, "exactly one"},
		{"unknown preset", []string{"check", "--preset", "prod"}, 2, "unknown preset"},
		{"unexpected argument", []string{"check", "-f", blocked, "10.0.0.0/8"}, 2, "unexpected argument"},
		{"unknown flag", []string{"check", "--bogus"}, 2, ""},
		{"bad concurrency", []string{"check", "-f", blocked, "--concurrency", "0"}, 2, "concurrency"},
		{"too much concurrency", []string{"check", "-f", blocked, "--concurrency", "1000"}, 2, "concurrency"},
		{"negative timeout", []string{"check", "-f", blocked, "--timeout", "-1s"}, 2, "positive"},
		{"missing file", []string{"check", "-f", filepath.Join(t.TempDir(), "nope.yaml")}, 2, "FAIL"},
		{"CIDR target is bad input", []string{"check", "-f", cidr}, 2, "CIDR"},
		{"all expectations met", []string{"check", "-f", blocked}, 0, "PASS"},
		{"unmet error-severity check", []string{"check", "-f", failing}, 1, "FAIL"},
		{"unmet warn-severity check", []string{"check", "-f", warnOnly}, 0, "WARN"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			code, out, errOut := net_(t, c.args...)
			if code != c.want {
				t.Fatalf("exit %d, want %d\nstdout: %s\nstderr: %s", code, c.want, out, errOut)
			}
			if !strings.Contains(out+errOut, c.in) {
				t.Errorf("output does not mention %q:\nstdout: %s\nstderr: %s", c.in, out, errOut)
			}
		})
	}
}

func TestNetCheckJSON(t *testing.T) {
	path := writeChecks(t, fmt.Sprintf("checks:\n  - {id: c, kind: tcp, target: \"127.0.0.1:%d\"}\n", closedLoopbackPort(t)))
	code, out, _ := net_(t, "check", "-f", path, "--json")
	if code != 1 {
		t.Fatalf("exit %d", code)
	}
	var rep struct {
		Summary struct{ Fail int } `json:"summary"`
		Results []struct{ ID, Status, Observed string }
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("stdout is not JSON: %v\n%s", err, out)
	}
	if rep.Summary.Fail != 1 || len(rep.Results) != 1 || rep.Results[0].Status != "fail" || rep.Results[0].Observed != "blocked" {
		t.Fatalf("%+v", rep)
	}
}

func TestNetCheckLocalPresetNeverFails(t *testing.T) {
	code, out, errOut := net_(t, "check", "--preset", "local")
	if code != 0 {
		t.Fatalf("exit %d: the local preset must not fail when services are down\n%s%s", code, out, errOut)
	}
	if !strings.Contains(out, "local-api") || !strings.Contains(out, "local-gpu-exporter") {
		t.Errorf("preset endpoints missing from output:\n%s", out)
	}
}
