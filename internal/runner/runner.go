// Package runner implements `forgectl run`: validate a TrainingJob, launch the Python worker for it, stream the
// worker's JSON-lines events, forward stop signals and map the worker's exit code (spec/worker-contract.md).
package runner

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"tinyforge.dev/forgectl/internal/jobspec"
	"tinyforge.dev/forgectl/internal/plan"
)

// Options configures one run. Everything that touches the machine is injectable so tests start no real process.
type Options struct {
	SpecPath string
	Out      string // worker --out dir; default runs/<metadata.name>
	Python   string // --python
	DryRun   bool   // pass --dry-run to the worker
	JSON     bool   // pass the worker's raw stdout lines through unchanged and keep stdout pure

	Stdout, Stderr io.Writer
	Signals        <-chan os.Signal // stop requests (Ctrl-C, SIGTERM); nil means none
	Grace          time.Duration    // after the first signal, wait this long before killing the worker (default 30s)

	Launch   Launcher
	Now      func() time.Time
	Getenv   func(string) string
	Cwd      string
	Exists   func(string) bool
	LookPath func(string) (string, error)
}

func (o *Options) defaults() {
	if o.Stdout == nil {
		o.Stdout = io.Discard
	}
	if o.Stderr == nil {
		o.Stderr = io.Discard
	}
	if o.Grace <= 0 {
		o.Grace = 30 * time.Second
	}
	if o.Launch == nil {
		o.Launch = ExecLauncher
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	if o.Getenv == nil {
		o.Getenv = os.Getenv
	}
	if o.Cwd == "" {
		o.Cwd, _ = os.Getwd()
	}
	if o.Exists == nil {
		o.Exists = func(p string) bool { st, err := os.Stat(p); return err == nil && !st.IsDir() }
	}
	if o.LookPath == nil {
		o.LookPath = exec.LookPath
	}
}

// syncWriter serialises writes: the event loop and the signal goroutine both print.
type syncWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (s *syncWriter) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.w.Write(p)
}

// Run returns the process exit code: 2 for an invalid spec, otherwise the worker's own exit code
// (1 if the worker could not be started or was killed).
func Run(o Options) int {
	o.defaults()
	stdout := &syncWriter{w: o.Stdout}
	stderr := &syncWriter{w: o.Stderr}

	job, err := jobspec.Load(o.SpecPath)
	if err != nil {
		fmt.Fprintln(stderr, "FAIL:", err)
		return ExitSpec
	}
	for _, w := range plan.Build(job).Warnings {
		fmt.Fprintln(stderr, "WARN:", w)
	}

	out := o.Out
	if out == "" {
		out = filepath.Join("runs", job.Metadata.Name)
	}
	python, source, err := FindPython(o.Python, o.Getenv, o.Cwd, o.Exists, o.LookPath)
	if err != nil {
		fmt.Fprintln(stderr, "FAIL:", err)
		return ExitCrash
	}
	args := []string{"-m", "tinyforge", "worker", "run", "--spec", o.SpecPath, "--out", out}
	if o.DryRun {
		args = append(args, "--dry-run")
	}
	fmt.Fprintf(stderr, "forgectl: running %s %s (python from %s)\n", python, strings.Join(args, " "), source)

	child, err := o.Launch(python, args, stderr)
	if err != nil {
		fmt.Fprintln(stderr, "FAIL: could not start the worker:", err)
		return ExitCrash
	}

	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		watchSignals(o, child, stderr, done)
	}()

	start := o.Now()
	streamEvents(o, child.Stdout(), stdout, start)
	code, werr := child.Wait()
	close(done)
	wg.Wait()

	if werr != nil {
		fmt.Fprintln(stderr, "FAIL: waiting for the worker:", werr)
		return ExitCrash
	}
	if code < 0 { // killed by a signal: no exit code to pass on
		fmt.Fprintln(stderr, "forgectl: the worker was killed before it could exit (no checkpoint on kill; resume from the last periodic one)")
		return ExitCrash
	}
	summary := Describe(code, o.DryRun)
	if o.JSON {
		fmt.Fprintln(stderr, "forgectl:", summary) // keep stdout pure JSON lines
	} else {
		fmt.Fprintln(stdout, "forgectl:", summary)
	}
	return code
}

// streamEvents reads the worker's stdout to EOF. Garbage and unknown events are ignored in human mode and passed
// through untouched in --json mode.
func streamEvents(o Options, r io.Reader, stdout io.Writer, start time.Time) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if o.JSON {
			stdout.Write(append(append([]byte(nil), line...), '\n'))
			continue
		}
		if text, ok := RenderEvent(line, o.Now().Sub(start)); ok {
			fmt.Fprintln(stdout, text)
		} else if o.DryRun && looksLikeObject(line) {
			// --dry-run prints the resolved worker config as one JSON object without an "event" key.
			fmt.Fprintf(stdout, "resolved worker config: %s\n", strings.TrimSpace(string(line)))
		}
	}
}

func looksLikeObject(b []byte) bool {
	s := strings.TrimSpace(string(b))
	return strings.HasPrefix(s, "{") && strings.HasSuffix(s, "}")
}

// watchSignals forwards the first stop request gracefully, then kills the worker if it has not exited within
// Grace, or immediately on a second request.
func watchSignals(o Options, child Child, stderr io.Writer, done <-chan struct{}) {
	var grace <-chan time.Time
	n := 0
	for {
		select {
		case <-done:
			return
		case sig := <-o.Signals:
			n++
			if n == 1 {
				fmt.Fprintf(stderr, "forgectl: %v received: asking the worker to checkpoint and stop (again to kill)\n", sig)
				if err := child.Forward(sig); err != nil {
					fmt.Fprintln(stderr, "forgectl: could not forward the signal:", err)
				}
				grace = time.After(o.Grace)
			} else {
				fmt.Fprintln(stderr, "forgectl: killing the worker")
				child.Kill()
			}
		case <-grace:
			grace = nil
			fmt.Fprintf(stderr, "forgectl: worker did not stop within %v: killing it\n", o.Grace)
			child.Kill()
		}
	}
}
