package runner

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestMain doubles as the "worker" for the one test that exercises the real exec launcher: the test binary re-executes
// itself with RUNNER_HELPER set, prints a few event lines and exits with a chosen code. No Python is ever started.
func TestMain(m *testing.M) {
	if mode := os.Getenv("RUNNER_HELPER"); mode != "" {
		fmt.Println("not json at all")
		fmt.Println(`{"event":"started","params":1234,"device":"cpu"}`)
		fmt.Println(`{"event":"step","step":5,"loss":2.5}`)
		fmt.Fprintln(os.Stderr, "helper stderr line")
		if mode == "exit3" {
			os.Exit(3)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

const minimalSpec = `
apiVersion: tinyforge.dev/v1alpha1
kind: TrainingJob
metadata: {name: t1}
spec:
  method: lora
  model: {base: org/model, revision: abc123}
  data: {source: org/data}
`

func writeSpec(t *testing.T, doc string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "job.yaml")
	if err := os.WriteFile(p, []byte(doc), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

// ---------------------------------------------------------------- fake child

type fakeChild struct {
	pr        *io.PipeReader
	pw        *io.PipeWriter
	mu        sync.Mutex
	forwarded []os.Signal
	kills     int
	exit      int
	waitCh    chan struct{}
	once      sync.Once
	onForward func(f *fakeChild)
}

func newFake() *fakeChild {
	pr, pw := io.Pipe()
	return &fakeChild{pr: pr, pw: pw, waitCh: make(chan struct{})}
}

// script writes the lines, then (if code >= -1 && autoFinish) exits with code.
func (f *fakeChild) script(lines []string, autoFinish bool, code int) *fakeChild {
	go func() {
		for _, l := range lines {
			if _, err := io.WriteString(f.pw, l+"\n"); err != nil {
				return
			}
		}
		if autoFinish {
			f.finish(code)
		}
	}()
	return f
}

func (f *fakeChild) finish(code int) {
	f.once.Do(func() {
		f.exit = code
		f.pw.Close()
		close(f.waitCh)
	})
}

func (f *fakeChild) Stdout() io.Reader { return f.pr }
func (f *fakeChild) Wait() (int, error) {
	<-f.waitCh
	return f.exit, nil
}
func (f *fakeChild) Forward(sig os.Signal) error {
	f.mu.Lock()
	f.forwarded = append(f.forwarded, sig)
	cb := f.onForward
	f.mu.Unlock()
	if cb != nil {
		cb(f)
	}
	return nil
}
func (f *fakeChild) Kill() error {
	f.mu.Lock()
	f.kills++
	f.mu.Unlock()
	f.finish(-1)
	return nil
}
func (f *fakeChild) counts() (forwarded, kills int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.forwarded), f.kills
}

type launched struct {
	name string
	args []string
}

func launcherFor(child Child, rec *launched) Launcher {
	return func(name string, args []string, _ io.Writer) (Child, error) {
		if rec != nil {
			rec.name, rec.args = name, args
		}
		return child, nil
	}
}

type result struct {
	code           int
	stdout, stderr string
}

func run(t *testing.T, o Options) result {
	t.Helper()
	var so, se bytes.Buffer
	o.Stdout, o.Stderr = &so, &se
	if o.Python == "" {
		o.Python = "fakepython"
	}
	code := Run(o)
	return result{code, so.String(), se.String()}
}

// ---------------------------------------------------------------- rendering

func TestRenderEvent(t *testing.T) {
	cases := []struct {
		name, line string
		want       []string // substrings that must all appear
		ok         bool
	}{
		{"started", `{"event":"started","params":1234,"device":"cuda","quant":"none"}`,
			[]string{"started", "params=1234", "device=cuda", "quant=none"}, true},
		{"resumed", `{"event":"resumed","step":50}`, []string{"resumed from step 50"}, true},
		{"step", `{"event":"step","step":25,"loss":1.23456,"lr":0.0002,"tok_per_s":1258.4,"peak_mem_gb":2.95}`,
			[]string{"step 25", "loss=1.235", "lr=0.0002", "tok_per_s=1258", "peak_mem_gb=2.95"}, true},
		{"eval", `{"event":"eval","step":50,"val_loss":1.3,"val_ppl":3.67}`,
			[]string{"eval step 50", "val_loss=1.3", "val_ppl=3.67"}, true},
		{"diagnostic with fix", `{"event":"diagnostic","code":"FT005","level":"error","message":"spill","fix":"lower max-len"}`,
			[]string{"[error] FT005 spill", "fix: lower max-len"}, true},
		{"diagnostic without fix", `{"event":"diagnostic","code":"FT004","level":"info","message":"ok","fix":""}`,
			[]string{"[info] FT004 ok"}, true},
		{"gate pass", `{"event":"gate","metric":"val_loss","op":"<","value":1.5,"observed":1.2,"passed":true}`,
			[]string{"gate val_loss < 1.5: PASS (observed 1.2)"}, true},
		{"gate fail", `{"event":"gate","metric":"val_loss","op":"<","value":1,"observed":1.2,"passed":false}`,
			[]string{"FAIL (observed 1.2)"}, true},
		{"gate unevaluable", `{"event":"gate","metric":"gain_pct","op":">=","value":5,"observed":null,"passed":false}`,
			[]string{"FAIL (could not be evaluated)"}, true},
		{"finished", `{"event":"finished","steps":240,"best_val_loss":0.187,"peak_mem_gb":2.4}`,
			[]string{"finished", "steps=240", "best_val_loss=0.187"}, true},
		{"failed", `{"event":"failed","reason":"out of memory"}`, []string{"FAILED: out of memory"}, true},
		{"failed without reason", `{"event":"failed"}`, []string{"FAILED: -"}, true},
		{"unknown event is ignored", `{"event":"heartbeat","t":1}`, nil, false},
		{"future event with fields is ignored", `{"event":"artifact","kind":"adapter","path":"x","sha256":"y"}`, nil, false},
		{"no event key", `{"lr":1}`, nil, false},
		{"event is not a string", `{"event":5}`, nil, false},
		{"plain text", `loading model...`, nil, false},
		{"truncated json", `{"event":"step","step":`, nil, false},
		{"json array", `[1,2,3]`, nil, false},
		{"empty", ``, nil, false},
	}
	for _, c := range cases {
		got, ok := RenderEvent([]byte(c.line), 75*time.Second)
		if ok != c.ok {
			t.Errorf("%s: ok=%v want %v (text %q)", c.name, ok, c.ok, got)
			continue
		}
		for _, w := range c.want {
			if !strings.Contains(got, w) {
				t.Errorf("%s: %q missing %q", c.name, got, w)
			}
		}
		if ok && !strings.HasPrefix(got, "[01:15] ") {
			t.Errorf("%s: missing elapsed prefix: %q", c.name, got)
		}
	}
}

func TestDescribe(t *testing.T) {
	cases := []struct {
		code   int
		dry    bool
		substr string
	}{
		{0, false, "succeeded"}, {0, true, "dry run OK"}, {1, false, "crashed"}, {2, false, "spec invalid"},
		{3, false, "gate failed"}, {4, false, "out of memory"}, {5, false, "diverged"},
		{75, false, "preempted"}, {7, false, "code 7"},
	}
	for _, c := range cases {
		if got := Describe(c.code, c.dry); !strings.Contains(got, c.substr) {
			t.Errorf("Describe(%d,%v) = %q, want substring %q", c.code, c.dry, got, c.substr)
		}
	}
}

// ---------------------------------------------------------------- python discovery

func TestFindPythonOrder(t *testing.T) {
	root := filepath.Join(string(filepath.Separator), "proj")
	sub := filepath.Join(root, "a", "b")
	venvWin := filepath.Join(root, ".venv", "Scripts", "python.exe")
	venvNix := filepath.Join(root, ".venv", "bin", "python")
	none := func(string) string { return "" }
	envSet := func(k string) string {
		if k == "FORGECTL_PYTHON" {
			return "/env/python"
		}
		return ""
	}
	existsOnly := func(paths ...string) func(string) bool {
		return func(p string) bool {
			for _, x := range paths {
				if x == p {
					return true
				}
			}
			return false
		}
	}
	pathOK := func(string) (string, error) { return "/usr/bin/python", nil }
	pathNo := func(string) (string, error) { return "", errors.New("not found") }

	cases := []struct {
		name         string
		explicit     string
		getenv       func(string) string
		exists       func(string) bool
		lookPath     func(string) (string, error)
		want, source string
		wantErr      bool
	}{
		{"flag beats everything", "/flag/python", envSet, existsOnly(venvWin), pathOK, "/flag/python", "--python", false},
		{"env beats venv and PATH", "", envSet, existsOnly(venvWin), pathOK, "/env/python", "FORGECTL_PYTHON", false},
		{"venv Scripts found walking up", "", none, existsOnly(venvWin), pathOK, venvWin, "project venv", false},
		{"venv bin found walking up", "", none, existsOnly(venvNix), pathOK, venvNix, "project venv", false},
		{"venv beats PATH", "", none, existsOnly(venvNix), pathNo, venvNix, "project venv", false},
		{"falls back to PATH", "", none, existsOnly(), pathOK, "/usr/bin/python", "PATH", false},
		{"nothing found", "", none, existsOnly(), pathNo, "", "", true},
	}
	for _, c := range cases {
		got, src, err := FindPython(c.explicit, c.getenv, sub, c.exists, c.lookPath)
		if (err != nil) != c.wantErr {
			t.Errorf("%s: err=%v wantErr=%v", c.name, err, c.wantErr)
			continue
		}
		if got != c.want || src != c.source {
			t.Errorf("%s: got (%q,%q) want (%q,%q)", c.name, got, src, c.want, c.source)
		}
	}
}

// ---------------------------------------------------------------- Run

func TestRunHumanOutputIgnoresGarbageAndUnknownEvents(t *testing.T) {
	child := newFake().script([]string{
		"loading model...",
		`{"event":"started","params":10,"device":"cpu"}`,
		`{"event":"heartbeat","t":1}`,
		`{"event":"step","step":5,"loss":2.5}`,
		`{"event":"finished","steps":5}`,
	}, true, 0)
	r := run(t, Options{SpecPath: writeSpec(t, minimalSpec), Launch: launcherFor(child, nil)})
	if r.code != 0 {
		t.Fatalf("code = %d, stderr: %s", r.code, r.stderr)
	}
	for _, w := range []string{"started params=10", "step 5 loss=2.5", "finished steps=5", "forgectl: succeeded"} {
		if !strings.Contains(r.stdout, w) {
			t.Errorf("stdout missing %q:\n%s", w, r.stdout)
		}
	}
	for _, bad := range []string{"loading model", "heartbeat"} {
		if strings.Contains(r.stdout, bad) {
			t.Errorf("stdout should not contain %q:\n%s", bad, r.stdout)
		}
	}
}

func TestRunJSONPassesRawLinesThroughUnchanged(t *testing.T) {
	lines := []string{"loading model...", `{"event":"step","step":5,"loss":2.5}`, `{"event":"heartbeat","t":1}`}
	child := newFake().script(lines, true, 0)
	r := run(t, Options{SpecPath: writeSpec(t, minimalSpec), JSON: true, Launch: launcherFor(child, nil)})
	if want := strings.Join(lines, "\n") + "\n"; r.stdout != want {
		t.Fatalf("stdout not pass-through.\n got: %q\nwant: %q", r.stdout, want)
	}
	if !strings.Contains(r.stderr, "succeeded") || strings.Contains(r.stdout, "forgectl:") {
		t.Errorf("summary must go to stderr in --json mode.\nstdout: %q\nstderr: %q", r.stdout, r.stderr)
	}
}

func TestRunExitCodeMapping(t *testing.T) {
	cases := []struct {
		workerCode, want int
		substr           string
	}{
		{0, 0, "succeeded"}, {2, 2, "spec invalid"}, {3, 3, "gate failed"}, {4, 4, "out of memory"},
		{5, 5, "diverged"}, {75, 75, "preempted"}, {1, 1, "crashed"}, {9, 9, "code 9"},
		{-1, 1, "killed"}, // killed by a signal: no code to pass on
	}
	for _, c := range cases {
		child := newFake().script([]string{`{"event":"failed","reason":"x"}`}, true, c.workerCode)
		r := run(t, Options{SpecPath: writeSpec(t, minimalSpec), Launch: launcherFor(child, nil)})
		if r.code != c.want {
			t.Errorf("worker %d: forgectl exit %d, want %d", c.workerCode, r.code, c.want)
		}
		if !strings.Contains(r.stdout+r.stderr, c.substr) {
			t.Errorf("worker %d: output missing %q:\n%s%s", c.workerCode, c.substr, r.stdout, r.stderr)
		}
	}
}

func TestRunInvalidSpecNeverLaunches(t *testing.T) {
	called := false
	launch := func(string, []string, io.Writer) (Child, error) { called = true; return nil, errors.New("no") }
	r := run(t, Options{SpecPath: writeSpec(t, strings.Replace(minimalSpec, "lora", "lora2", 1)), Launch: launch})
	if r.code != 2 || called || !strings.Contains(r.stderr, "FAIL") {
		t.Fatalf("code=%d launched=%v stderr=%s", r.code, called, r.stderr)
	}
	if r := run(t, Options{SpecPath: filepath.Join(t.TempDir(), "missing.yaml"), Launch: launch}); r.code != 2 || called {
		t.Fatalf("missing file: code=%d launched=%v", r.code, called)
	}
}

func TestRunLaunchArgsAndOutDefault(t *testing.T) {
	spec := writeSpec(t, minimalSpec)
	cases := []struct {
		name string
		o    Options
		want []string
	}{
		{"default out", Options{}, []string{"-m", "tinyforge", "worker", "run", "--spec", spec, "--out", filepath.Join("runs", "t1")}},
		{"explicit out", Options{Out: "/data/out"}, []string{"-m", "tinyforge", "worker", "run", "--spec", spec, "--out", "/data/out"}},
		{"dry run", Options{DryRun: true}, []string{"-m", "tinyforge", "worker", "run", "--spec", spec, "--out", filepath.Join("runs", "t1"), "--dry-run"}},
	}
	for _, c := range cases {
		var rec launched
		c.o.SpecPath = spec
		c.o.Python = "mypython"
		c.o.Launch = launcherFor(newFake().script(nil, true, 0), &rec)
		if r := run(t, c.o); r.code != 0 {
			t.Fatalf("%s: code %d: %s", c.name, r.code, r.stderr)
		}
		if rec.name != "mypython" || strings.Join(rec.args, "|") != strings.Join(c.want, "|") {
			t.Errorf("%s: launched %s %v\nwant %v", c.name, rec.name, rec.args, c.want)
		}
	}
}

func TestRunPrintsPlanWarningsToStderr(t *testing.T) {
	unpinned := strings.Replace(minimalSpec, ", revision: abc123", "", 1)
	r := run(t, Options{SpecPath: writeSpec(t, unpinned), Launch: launcherFor(newFake().script(nil, true, 0), nil)})
	if !strings.Contains(r.stderr, "WARN:") || !strings.Contains(r.stderr, "not pinned") || strings.Contains(r.stdout, "WARN") {
		t.Fatalf("warnings must be on stderr.\nstdout: %q\nstderr: %q", r.stdout, r.stderr)
	}
}

func TestRunDryRunShowsResolvedConfig(t *testing.T) {
	child := newFake().script([]string{`{"max_steps":240,"lr":0.0002,"quant":"none"}`}, true, 0)
	r := run(t, Options{SpecPath: writeSpec(t, minimalSpec), DryRun: true, Launch: launcherFor(child, nil)})
	if r.code != 0 || !strings.Contains(r.stdout, `resolved worker config: {"max_steps":240`) || !strings.Contains(r.stdout, "dry run OK") {
		t.Fatalf("code=%d\nstdout: %s\nstderr: %s", r.code, r.stdout, r.stderr)
	}
	// Outside --dry-run an event-less object is just noise and must be ignored.
	child = newFake().script([]string{`{"max_steps":240}`}, true, 0)
	if r := run(t, Options{SpecPath: writeSpec(t, minimalSpec), Launch: launcherFor(child, nil)}); strings.Contains(r.stdout, "resolved") {
		t.Errorf("event-less object printed outside --dry-run: %s", r.stdout)
	}
}

func TestRunLaunchFailureAndNoPython(t *testing.T) {
	boom := func(string, []string, io.Writer) (Child, error) { return nil, errors.New("exec: not found") }
	if r := run(t, Options{SpecPath: writeSpec(t, minimalSpec), Launch: boom}); r.code != 1 || !strings.Contains(r.stderr, "could not start") {
		t.Errorf("launch failure: code=%d stderr=%s", r.code, r.stderr)
	}
	var so, se bytes.Buffer
	code := Run(Options{
		SpecPath: writeSpec(t, minimalSpec), Stdout: &so, Stderr: &se, Cwd: t.TempDir(),
		Getenv: func(string) string { return "" }, Exists: func(string) bool { return false },
		LookPath: func(string) (string, error) { return "", errors.New("nope") }, Launch: boom,
	})
	if code != 1 || !strings.Contains(se.String(), "no Python found") {
		t.Errorf("no python: code=%d stderr=%s", code, se.String())
	}
}

func TestRunElapsedUsesInjectedClock(t *testing.T) {
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	calls := 0
	now := func() time.Time {
		calls++
		if calls == 1 {
			return base
		}
		return base.Add(125 * time.Second)
	}
	child := newFake().script([]string{`{"event":"step","step":1,"loss":3}`}, true, 0)
	r := run(t, Options{SpecPath: writeSpec(t, minimalSpec), Now: now, Launch: launcherFor(child, nil)})
	if !strings.Contains(r.stdout, "[02:05] step 1") {
		t.Fatalf("elapsed prefix wrong:\n%s", r.stdout)
	}
}

// ---------------------------------------------------------------- signals

func TestSignalIsForwardedAndWorkerExitCodePropagates(t *testing.T) {
	child := newFake().script([]string{`{"event":"started","params":1}`}, false, 0)
	child.onForward = func(f *fakeChild) {
		go func() { // a cooperative worker: finishes the step, reports preemption, exits 75
			io.WriteString(f.pw, `{"event":"failed","reason":"preempted"}`+"\n")
			f.finish(75)
		}()
	}
	sigs := make(chan os.Signal, 2)
	sigs <- os.Interrupt
	r := run(t, Options{SpecPath: writeSpec(t, minimalSpec), Signals: sigs, Grace: time.Minute, Launch: launcherFor(child, nil)})
	fwd, kills := child.counts()
	if fwd != 1 || kills != 0 {
		t.Errorf("forwarded=%d kills=%d, want 1 and 0", fwd, kills)
	}
	if r.code != 75 || !strings.Contains(r.stdout, "FAILED: preempted") || !strings.Contains(r.stdout, "preempted (exit 75)") {
		t.Errorf("code=%d\nstdout: %s\nstderr: %s", r.code, r.stdout, r.stderr)
	}
	if !strings.Contains(r.stderr, "asking the worker to checkpoint") {
		t.Errorf("stderr should explain the forward: %s", r.stderr)
	}
}

func TestWorkerThatIgnoresSignalIsKilledAfterGrace(t *testing.T) {
	child := newFake().script([]string{`{"event":"started","params":1}`}, false, 0) // never exits on Forward
	sigs := make(chan os.Signal, 1)
	sigs <- os.Interrupt
	r := run(t, Options{SpecPath: writeSpec(t, minimalSpec), Signals: sigs, Grace: 30 * time.Millisecond, Launch: launcherFor(child, nil)})
	fwd, kills := child.counts()
	if fwd != 1 || kills != 1 {
		t.Errorf("forwarded=%d kills=%d, want 1 and 1", fwd, kills)
	}
	if r.code != 1 || !strings.Contains(r.stderr, "did not stop within") {
		t.Errorf("code=%d stderr: %s", r.code, r.stderr)
	}
}

func TestSecondSignalKillsImmediately(t *testing.T) {
	child := newFake().script([]string{`{"event":"started","params":1}`}, false, 0)
	sigs := make(chan os.Signal, 2)
	sigs <- os.Interrupt
	sigs <- os.Interrupt
	r := run(t, Options{SpecPath: writeSpec(t, minimalSpec), Signals: sigs, Grace: time.Hour, Launch: launcherFor(child, nil)})
	fwd, kills := child.counts()
	if fwd != 1 || kills != 1 || r.code != 1 || !strings.Contains(r.stderr, "killing the worker") {
		t.Errorf("forwarded=%d kills=%d code=%d stderr: %s", fwd, kills, r.code, r.stderr)
	}
}

// ---------------------------------------------------------------- real exec launcher (test binary as the "worker")

func TestExecLauncherStreamsRealChildAndPropagatesExitCode(t *testing.T) {
	for _, c := range []struct {
		mode string
		want int
	}{{"exit0", 0}, {"exit3", 3}} {
		t.Setenv("RUNNER_HELPER", c.mode)
		r := run(t, Options{SpecPath: writeSpec(t, minimalSpec), Python: os.Args[0], Launch: ExecLauncher})
		if r.code != c.want {
			t.Errorf("%s: exit %d want %d\nstdout: %s\nstderr: %s", c.mode, r.code, c.want, r.stdout, r.stderr)
		}
		if !strings.Contains(r.stdout, "started params=1234 device=cpu") || !strings.Contains(r.stdout, "step 5 loss=2.5") {
			t.Errorf("%s: events not streamed:\n%s", c.mode, r.stdout)
		}
		if strings.Contains(r.stdout, "not json at all") {
			t.Errorf("%s: garbage leaked into human output", c.mode)
		}
		if !strings.Contains(r.stderr, "helper stderr line") {
			t.Errorf("%s: child stderr not forwarded:\n%s", c.mode, r.stderr)
		}
	}
}
