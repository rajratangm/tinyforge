package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"tinyforge.dev/forgectl/internal/runner"
)

// ---------------------------------------------------------------- a fake worker: no test ever starts Python

// gate is a one-shot latch a test opens to let a fake worker finish.
type gate struct {
	once sync.Once
	ch   chan struct{}
}

func newGate() *gate  { return &gate{ch: make(chan struct{})} }
func (g *gate) open() { g.once.Do(func() { close(g.ch) }) }

// script describes one worker run. The worker prints lines, then exits with code once hold (if any) opens.
// If onTerm is set the worker exits with that code as soon as it is asked to stop (SIGTERM); otherwise it ignores
// the request and only a kill ends it.
type script struct {
	lines  []string
	hold   *gate
	code   int
	onTerm *int
}

type launched struct{ name, spec, out, jobID string }

type fakeLauncher struct {
	mu         sync.Mutex
	scripts    []script
	launches   []launched
	children   []*fakeChild
	holds      []*gate
	running    int
	maxRunning int
}

func (f *fakeLauncher) add(sc ...script) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, s := range sc {
		if s.hold != nil {
			f.holds = append(f.holds, s.hold)
		}
	}
	f.scripts = append(f.scripts, sc...)
}

func (f *fakeLauncher) releaseAll() {
	f.mu.Lock()
	hs := append([]*gate(nil), f.holds...)
	f.mu.Unlock()
	for _, h := range hs {
		h.open()
	}
}

// waitChild returns the i-th launched worker, waiting for the launch (a job is marked running just before it).
func (f *fakeLauncher) waitChild(t *testing.T, i int) *fakeChild {
	t.Helper()
	waitFor(t, "the worker to be launched", func() bool {
		f.mu.Lock()
		defer f.mu.Unlock()
		return len(f.children) > i
	})
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.children[i]
}

func (f *fakeLauncher) launchCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.launches)
}

func (f *fakeLauncher) launch(name string, args []string, stderr io.Writer) (runner.Child, error) {
	f.mu.Lock()
	if len(f.scripts) == 0 {
		f.mu.Unlock()
		return nil, errors.New("fakeLauncher: no script queued for this launch")
	}
	sc := f.scripts[0]
	f.scripts = f.scripts[1:]
	l := launched{name: name}
	for i, a := range args {
		switch a {
		case "--spec":
			l.spec = args[i+1]
			l.jobID = filepath.Base(filepath.Dir(l.spec))
		case "--out":
			l.out = args[i+1]
		}
	}
	f.launches = append(f.launches, l)
	f.running++
	if f.running > f.maxRunning {
		f.maxRunning = f.running
	}
	c := &fakeChild{f: f, sc: sc, stop: make(chan int, 2), done: make(chan int, 1)}
	c.pr, c.pw = io.Pipe()
	f.children = append(f.children, c)
	f.mu.Unlock()
	fmt.Fprintln(stderr, "fake worker started")
	go c.run()
	return c, nil
}

type fakeChild struct {
	f      *fakeLauncher
	sc     script
	pr     *io.PipeReader
	pw     *io.PipeWriter
	stop   chan int
	done   chan int
	once   sync.Once
	mu     sync.Mutex
	sigs   []os.Signal
	killed bool
}

func (c *fakeChild) run() {
	for _, l := range c.sc.lines {
		fmt.Fprintln(c.pw, l)
	}
	if c.sc.hold == nil {
		c.finish(c.sc.code)
		return
	}
	select {
	case <-c.sc.hold.ch:
		c.finish(c.sc.code)
	case code := <-c.stop:
		c.finish(code)
	}
}

func (c *fakeChild) finish(code int) {
	c.once.Do(func() {
		_ = c.pw.Close()
		c.f.mu.Lock()
		c.f.running--
		c.f.mu.Unlock()
		c.done <- code
	})
}

func (c *fakeChild) Stdout() io.Reader { return c.pr }
func (c *fakeChild) Wait() (int, error) {
	return <-c.done, nil
}
func (c *fakeChild) Forward(sig os.Signal) error {
	c.mu.Lock()
	c.sigs = append(c.sigs, sig)
	c.mu.Unlock()
	if c.sc.onTerm != nil {
		select {
		case c.stop <- *c.sc.onTerm:
		default:
		}
	}
	return nil
}
func (c *fakeChild) Kill() error {
	c.mu.Lock()
	c.killed = true
	c.mu.Unlock()
	select {
	case c.stop <- -1:
	default:
	}
	return nil
}
func (c *fakeChild) wasKilled() bool { c.mu.Lock(); defer c.mu.Unlock(); return c.killed }
func (c *fakeChild) signals() int    { c.mu.Lock(); defer c.mu.Unlock(); return len(c.sigs) }

func ip(v int) *int { return &v }

// ---------------------------------------------------------------- helpers

func newExecServer(t *testing.T, dir string, mut func(*Config)) (*Server, *fakeLauncher) {
	t.Helper()
	fl := &fakeLauncher{}
	s, _ := newTestServer(t, func(c *Config) {
		c.StateDir, c.Execute, c.Python = dir, true, "fake-python"
		c.Launch = fl.launch
		c.CancelGrace = 150 * time.Millisecond
		c.MaxRetries = 2
		c.Now = time.Now
		if mut != nil {
			mut(c)
		}
	})
	return s, fl
}

func startExec(t *testing.T, s *Server, fl *fakeLauncher) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	stop := s.StartExecutor(ctx)
	t.Cleanup(func() { stop(); cancel() }) // cleanups run last-in first-out: workers are released before stop waits
	t.Cleanup(fl.releaseAll)
}

func submit(t *testing.T, s *Server) string {
	t.Helper()
	rec := do(s, "POST", "/v1/jobs", validSpec, nil)
	if rec.Code != 202 {
		t.Fatalf("submit = %d: %s", rec.Code, rec.Body.String())
	}
	var j Job
	if err := json.Unmarshal(rec.Body.Bytes(), &j); err != nil {
		t.Fatal(err)
	}
	return j.ID
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func jobOf(t *testing.T, s *Server, id string) Job {
	t.Helper()
	j, ok := s.jobs.get(id)
	if !ok {
		t.Fatalf("job %s not found", id)
	}
	return j
}

func waitStatus(t *testing.T, s *Server, id, want string) Job {
	t.Helper()
	waitFor(t, fmt.Sprintf("job %s to be %s (now %s)", id, want, jobOf(t, s, id).Status),
		func() bool { return jobOf(t, s, id).Status == want })
	return jobOf(t, s, id)
}

// events returns a job's events (replay only, no follow) as decoded objects.
func eventsOf(t *testing.T, s *Server, id string) []map[string]any {
	t.Helper()
	rec := do(s, "GET", "/v1/jobs/"+id+"/events?follow=0&tail=1000", "", nil)
	if rec.Code != 200 {
		t.Fatalf("events = %d: %s", rec.Code, rec.Body.String())
	}
	var out []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(rec.Body.String()), "\n") {
		if line == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("event line is not JSON: %q", line)
		}
		out = append(out, m)
	}
	return out
}

func kinds(evs []map[string]any) []string {
	var k []string
	for _, e := range evs {
		k = append(k, fmt.Sprint(e["event"]))
	}
	return k
}

const (
	evStarted  = `{"event":"started","t":1}`
	evStep     = `{"event":"step","t":2,"step":5,"loss":2.5}`
	evFinished = `{"event":"finished","t":3,"steps":5}`
)

// ---------------------------------------------------------------- lifecycle

func TestLifecycleSucceedsAndRunsTheWorkerOnItsOwnFiles(t *testing.T) {
	dir := t.TempDir()
	s, fl := newExecServer(t, dir, nil)
	fl.add(script{lines: []string{evStarted, evStep, "not json at all", evFinished}, code: 0})
	startExec(t, s, fl)
	id := submit(t, s)
	j := waitStatus(t, s, id, StatusSucceeded)

	if j.ExitCode == nil || *j.ExitCode != 0 || j.Attempts != 1 || j.Started == nil || j.Finished == nil {
		t.Fatalf("unexpected final job: %+v", j)
	}
	l := fl.launches[0]
	jobDir := filepath.Join(dir, "jobs", id)
	if l.name != "fake-python" || l.spec != filepath.Join(jobDir, "spec.yaml") || l.out != filepath.Join(jobDir, "out") {
		t.Fatalf("worker was launched with %+v, want python=fake-python spec/out inside %s", l, jobDir)
	}
	if b, _ := os.ReadFile(l.spec); !strings.Contains(string(b), "name: t1") {
		t.Fatalf("spec.yaml is not the submitted spec: %q", b)
	}
	got := kinds(eventsOf(t, s, id)) // eventsOf fails on any non-JSON line, so the garbage line must have been dropped
	want := []string{"job_started", "started", "step", "finished", "job_end"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("events = %v, want %v", got, want)
	}
	if b, _ := os.ReadFile(filepath.Join(jobDir, "worker.log")); !strings.Contains(string(b), "fake worker started") {
		t.Fatalf("worker stderr was not captured: %q", b)
	}
}

func TestFIFOOrderAndConcurrencyLimit(t *testing.T) {
	s, fl := newExecServer(t, t.TempDir(), nil) // MaxConcurrent defaults to 1
	gates := []*gate{newGate(), newGate(), newGate()}
	for _, g := range gates {
		fl.add(script{hold: g, code: 0})
	}
	startExec(t, s, fl)
	ids := []string{submit(t, s), submit(t, s), submit(t, s)}

	for i, id := range ids {
		waitStatus(t, s, id, StatusRunning)
		for k := i + 1; k < len(ids); k++ {
			if st := jobOf(t, s, ids[k]).Status; st != StatusQueued {
				t.Fatalf("job %d is %s while job %d runs: concurrency limit of 1 broken", k, st, i)
			}
		}
		gates[i].open()
		waitStatus(t, s, id, StatusSucceeded)
	}
	for i, id := range ids {
		if fl.launches[i].jobID != id {
			t.Fatalf("launch %d was %s, want %s: not oldest-first", i, fl.launches[i].jobID, id)
		}
	}
	if fl.maxRunning != 1 {
		t.Fatalf("max concurrent workers = %d, want 1", fl.maxRunning)
	}
}

func TestMaxConcurrentTwoRunsTwoAtOnce(t *testing.T) {
	s, fl := newExecServer(t, t.TempDir(), func(c *Config) { c.MaxConcurrent = 2 })
	g := newGate()
	fl.add(script{hold: g}, script{hold: g}, script{hold: g})
	startExec(t, s, fl)
	a, b, c := submit(t, s), submit(t, s), submit(t, s)
	waitStatus(t, s, a, StatusRunning)
	waitStatus(t, s, b, StatusRunning)
	time.Sleep(50 * time.Millisecond)
	if st := jobOf(t, s, c).Status; st != StatusQueued {
		t.Fatalf("third job is %s, want queued while two run", st)
	}
	g.open()
	waitStatus(t, s, c, StatusSucceeded)
	if fl.maxRunning != 2 {
		t.Fatalf("max concurrent = %d, want 2", fl.maxRunning)
	}
}

// ---------------------------------------------------------------- exit codes

func TestPreemptedIsRetriedThenSucceeds_RetryKeepsItsPlaceInLine(t *testing.T) {
	s, fl := newExecServer(t, t.TempDir(), nil)
	fl.add(script{lines: []string{evStarted}, code: 75}, script{lines: []string{evFinished}, code: 0}, script{code: 0})
	startExec(t, s, fl)
	a, b := submit(t, s), submit(t, s)
	ja := waitStatus(t, s, a, StatusSucceeded)
	waitStatus(t, s, b, StatusSucceeded)

	if ja.Attempts != 2 {
		t.Fatalf("attempts = %d, want 2", ja.Attempts)
	}
	order := []string{fl.launches[0].jobID, fl.launches[1].jobID, fl.launches[2].jobID}
	if strings.Join(order, ",") != strings.Join([]string{a, a, b}, ",") {
		t.Fatalf("launch order %v, want the preempted job to resume before the next one starts", order)
	}
	got := kinds(eventsOf(t, s, a))
	if !contains(got, "job_requeued") || got[len(got)-1] != "job_end" {
		t.Fatalf("events %v: want a job_requeued then job_end", got)
	}
}

func TestPreemptedGivesUpAfterMaxRetries(t *testing.T) {
	s, fl := newExecServer(t, t.TempDir(), func(c *Config) { c.MaxRetries = 1 })
	fl.add(script{code: 75}, script{code: 75}, script{code: 75})
	startExec(t, s, fl)
	id := submit(t, s)
	j := waitStatus(t, s, id, StatusFailed)
	if fl.launchCount() != 2 || j.Attempts != 2 || *j.ExitCode != 75 || !strings.Contains(j.Message, "giving up") {
		t.Fatalf("launches=%d job=%+v: want 2 attempts (1 retry) then failed", fl.launchCount(), j)
	}
}

func TestNoRetryOnContractFailures(t *testing.T) {
	cases := map[int]string{2: "spec invalid", 3: "gate failed", 4: "out of memory", 5: "diverged", 1: "crashed", 9: "code 9"}
	for code, want := range cases {
		t.Run(fmt.Sprintf("exit%d", code), func(t *testing.T) {
			s, fl := newExecServer(t, t.TempDir(), nil)
			fl.add(script{code: code}, script{code: 0}) // a retry would consume the second script and succeed
			startExec(t, s, fl)
			id := submit(t, s)
			j := waitStatus(t, s, id, StatusFailed)
			time.Sleep(60 * time.Millisecond)
			if fl.launchCount() != 1 || j.Attempts != 1 {
				t.Fatalf("exit %d was retried: launches=%d attempts=%d", code, fl.launchCount(), j.Attempts)
			}
			if j.ExitCode == nil || *j.ExitCode != code || !strings.Contains(j.Message, want) {
				t.Fatalf("job = %+v, want exit %d and a message containing %q", j, code, want)
			}
		})
	}
}

func TestDecideTable(t *testing.T) {
	type in struct {
		code            int
		cancel, shutdwn bool
		attempts, max   int
	}
	cases := []struct {
		name string
		in   in
		want string
	}{
		{"ok", in{0, false, false, 1, 2}, StatusSucceeded},
		{"ok even if cancel was requested", in{0, true, false, 1, 2}, StatusSucceeded},
		{"preempted, retries left", in{75, false, false, 2, 2}, StatusQueued},
		{"preempted, no retries left", in{75, false, false, 3, 2}, StatusFailed},
		{"preempted with retries disabled", in{75, false, false, 1, 0}, StatusFailed},
		{"cancel beats preempted", in{75, true, false, 1, 2}, StatusCancelled},
		{"cancel after crash", in{-1, true, false, 1, 2}, StatusCancelled},
		{"shutdown + preempted resumes later", in{75, false, true, 1, 0}, StatusQueued},
		{"shutdown + anything else is interrupted", in{1, false, true, 1, 2}, StatusInterrupted},
		{"spec", in{2, false, false, 1, 2}, StatusFailed},
		{"gate", in{3, false, false, 1, 2}, StatusFailed},
		{"fit", in{4, false, false, 1, 2}, StatusFailed},
		{"diverged", in{5, false, false, 1, 2}, StatusFailed},
	}
	for _, c := range cases {
		if got := decide(c.in.code, c.in.cancel, c.in.shutdwn, c.in.attempts, c.in.max); got.status != c.want || got.message == "" {
			t.Errorf("%s: status %q (message %q), want %q", c.name, got.status, got.message, c.want)
		}
	}
}

func TestWorkerNotFoundFailsClosedAndAgentStaysUp(t *testing.T) {
	s, fl := newExecServer(t, t.TempDir(), func(c *Config) {
		c.Python = ""
		c.Getenv = func(string) string { return "" }
		c.Cwd = t.TempDir()
		c.Exists = func(string) bool { return false }
		c.LookPath = func(string) (string, error) { return "", errors.New("not on PATH") }
	})
	startExec(t, s, fl)
	id := submit(t, s)
	j := waitStatus(t, s, id, StatusFailed)
	if !strings.Contains(j.Message, "AG001") || fl.launchCount() != 0 {
		t.Fatalf("job = %+v, launches = %d: want a clear AG001 failure and no launch", j, fl.launchCount())
	}
	evs := eventsOf(t, s, id)
	var diag map[string]any
	for _, e := range evs {
		if e["event"] == "diagnostic" {
			diag = e
		}
	}
	if diag == nil || diag["code"] != "AG001" || diag["level"] != "error" || diag["fix"] == "" {
		t.Fatalf("no AG001 diagnostic with a fix in the events: %v", evs)
	}
	if rec := do(s, "GET", "/healthz", "", nil); rec.Code != 200 {
		t.Fatalf("agent is not serving after a worker-not-found failure: %d", rec.Code)
	}
	// And it still accepts and fails further jobs the same way instead of wedging.
	id2 := submit(t, s)
	waitStatus(t, s, id2, StatusFailed)
}

// ---------------------------------------------------------------- restart recovery

func TestRestartRecovery(t *testing.T) {
	dir := t.TempDir()
	s1, _ := newExecServer(t, dir, func(c *Config) { c.Execute = false })
	a, b := submit(t, s1), submit(t, s1)
	// Simulate "the agent died while job a was running": claim it without ever starting a worker.
	if c, err := s1.jobs.claimNext(1); err != nil || c == nil || c.job.ID != a {
		t.Fatalf("claimNext = %v, %v", c, err)
	}

	s2, fl := newExecServer(t, dir, nil) // the restarted agent
	ja, jb := jobOf(t, s2, a), jobOf(t, s2, b)
	if ja.Status != StatusInterrupted || ja.Finished == nil || !strings.Contains(ja.Message, "not re-run") || ja.Attempts != 1 {
		t.Fatalf("running job after restart = %+v, want interrupted and not re-run", ja)
	}
	if jb.Status != StatusQueued {
		t.Fatalf("queued job after restart = %s, want queued", jb.Status)
	}
	if ids := []string{s2.jobs.list()[0].ID, s2.jobs.list()[1].ID}; ids[0] != a || ids[1] != b {
		t.Fatalf("order after restart = %v, want submission order", ids)
	}
	fl.add(script{lines: []string{evFinished}, code: 0})
	startExec(t, s2, fl)
	waitStatus(t, s2, b, StatusSucceeded)
	time.Sleep(60 * time.Millisecond)
	if fl.launchCount() != 1 || fl.launches[0].jobID != b {
		t.Fatalf("launches = %+v: only the queued job may run after a restart", fl.launches)
	}
	if st := jobOf(t, s2, a).Status; st != StatusInterrupted {
		t.Fatalf("interrupted job changed to %s", st)
	}
	// A third restart sees the same persisted truth.
	s3, _ := newExecServer(t, dir, func(c *Config) { c.Execute = false })
	if jobOf(t, s3, a).Status != StatusInterrupted || jobOf(t, s3, b).Status != StatusSucceeded {
		t.Fatalf("second restart disagrees with what was persisted")
	}
	if got := kinds(eventsOf(t, s3, b)); !contains(got, "finished") {
		t.Fatalf("events did not survive the restart: %v", got)
	}
}

// With a frozen clock every job has the same Submitted time: the queue order after a restart must still be the
// submission order (the persisted sequence number), not an accident of random ids.
func TestQueueOrderSurvivesRestartWithIdenticalTimestamps(t *testing.T) {
	dir := t.TempDir()
	s1, _ := newTestServer(t, func(c *Config) { c.StateDir = dir }) // newTestServer's clock never moves
	var want []string
	for i := 0; i < 8; i++ {
		want = append(want, submit(t, s1))
	}
	s2, _ := newTestServer(t, func(c *Config) { c.StateDir = dir })
	var got []string
	for _, j := range s2.jobs.list() {
		got = append(got, j.ID)
	}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("order after restart = %v, want submission order %v", got, want)
	}
	// And new submissions continue the sequence instead of reusing numbers.
	next := submit(t, s2)
	if j, _ := s2.jobs.get(next); j.Seq != int64(len(want))+1 {
		t.Fatalf("next sequence number = %d, want %d", j.Seq, len(want)+1)
	}
}

func TestStateDirIsValidated(t *testing.T) {
	if _, err := New(Config{Execute: true}); err == nil || !strings.Contains(err.Error(), "state") {
		t.Fatalf("Execute without a state dir = %v, want an error naming the state directory", err)
	}
}

func TestLoadIgnoresStrayAndCorruptDirectories(t *testing.T) {
	dir := t.TempDir()
	jobs := filepath.Join(dir, "jobs")
	for _, d := range []string{"evil", "..", "job-zzzzzzzz", "job-0000000a"} {
		_ = os.MkdirAll(filepath.Join(jobs, d), 0o700)
	}
	_ = os.WriteFile(filepath.Join(jobs, "job-0000000a", "job.json"), []byte("{broken"), 0o600)
	_ = os.WriteFile(filepath.Join(jobs, "evil", "job.json"), []byte(`{"id":"evil","status":"queued"}`), 0o600)
	s, _ := newExecServer(t, dir, func(c *Config) { c.Execute = false })
	if n := s.jobs.count(); n != 0 {
		t.Fatalf("loaded %d jobs from stray or corrupt directories, want 0", n)
	}
}

// ---------------------------------------------------------------- cancel and delete

func TestCancelQueuedJob(t *testing.T) {
	s, fl := newExecServer(t, t.TempDir(), nil)
	hold := newGate()
	fl.add(script{hold: hold}) // only the first job will ever launch
	startExec(t, s, fl)
	first, second := submit(t, s), submit(t, s)
	waitStatus(t, s, first, StatusRunning)

	rec := do(s, "POST", "/v1/jobs/"+second+"/cancel", "", nil)
	var j Job
	_ = json.Unmarshal(rec.Body.Bytes(), &j)
	if rec.Code != 200 || j.Status != StatusCancelled || j.Finished == nil {
		t.Fatalf("cancel queued = %d %s", rec.Code, rec.Body.String())
	}
	hold.open()
	waitStatus(t, s, first, StatusSucceeded)
	time.Sleep(60 * time.Millisecond)
	if fl.launchCount() != 1 {
		t.Fatalf("a cancelled queued job was launched (%d launches)", fl.launchCount())
	}
	if got := kinds(eventsOf(t, s, second)); got[len(got)-1] != "job_end" {
		t.Fatalf("cancelled job events = %v, want a job_end", got)
	}
}

func TestCancelRunningJobStopsGracefully(t *testing.T) {
	s, fl := newExecServer(t, t.TempDir(), nil)
	fl.add(script{lines: []string{evStarted}, hold: newGate(), onTerm: ip(75)}) // checkpoints and exits 75 on SIGTERM
	startExec(t, s, fl)
	id := submit(t, s)
	waitStatus(t, s, id, StatusRunning)

	rec := do(s, "POST", "/v1/jobs/"+id+"/cancel", "", nil)
	if rec.Code != 202 {
		t.Fatalf("cancel running = %d, want 202 (stopping)", rec.Code)
	}
	j := waitStatus(t, s, id, StatusCancelled) // cancelled, NOT re-queued as a preempted job
	if j.ExitCode == nil || *j.ExitCode != 75 || fl.launchCount() != 1 {
		t.Fatalf("job = %+v launches = %d: a user cancel must not be retried", j, fl.launchCount())
	}
	if fl.children[0].signals() != 1 || fl.children[0].wasKilled() {
		t.Fatalf("signals=%d killed=%v: want one graceful stop request and no kill", fl.children[0].signals(), fl.children[0].wasKilled())
	}
	if rec = do(s, "POST", "/v1/jobs/"+id+"/cancel", "", nil); rec.Code != 409 {
		t.Fatalf("cancel of a finished job = %d, want 409", rec.Code)
	}
}

func TestCancelKillsAWorkerThatIgnoresTheRequest(t *testing.T) {
	s, fl := newExecServer(t, t.TempDir(), nil) // CancelGrace is 150 ms
	fl.add(script{hold: newGate()})             // ignores SIGTERM
	startExec(t, s, fl)
	id := submit(t, s)
	waitStatus(t, s, id, StatusRunning)
	do(s, "POST", "/v1/jobs/"+id+"/cancel", "", nil)
	j := waitStatus(t, s, id, StatusCancelled)
	if !fl.children[0].wasKilled() || j.ExitCode == nil || *j.ExitCode != runner.ExitCrash {
		t.Fatalf("killed=%v job=%+v: a worker that ignores the stop request must be killed after the grace period",
			fl.children[0].wasKilled(), j)
	}
}

func TestForceCancelKillsImmediately(t *testing.T) {
	s, fl := newExecServer(t, t.TempDir(), func(c *Config) { c.CancelGrace = time.Hour })
	fl.add(script{hold: newGate()})
	startExec(t, s, fl)
	id := submit(t, s)
	waitStatus(t, s, id, StatusRunning)
	do(s, "POST", "/v1/jobs/"+id+"/cancel", "", nil)
	child := fl.waitChild(t, 0)
	waitFor(t, "the graceful request to be delivered", func() bool { return child.signals() == 1 })
	if child.wasKilled() {
		t.Fatal("killed before force was requested")
	}
	do(s, "POST", "/v1/jobs/"+id+"/cancel?force=1", "", nil)
	waitStatus(t, s, id, StatusCancelled)
	if !child.wasKilled() {
		t.Fatal("force=1 did not kill the worker")
	}
}

func TestCancelAndDeleteErrors(t *testing.T) {
	s, _ := newTestServer(t, nil)
	for _, p := range []string{"job-deadbeef", "..%2F..%2Fetc", "job-ZZZZZZZZ", "x"} {
		if rec := do(s, "POST", "/v1/jobs/"+p+"/cancel", "", nil); rec.Code != 404 {
			t.Errorf("cancel %q = %d, want 404", p, rec.Code)
		}
		if rec := do(s, "DELETE", "/v1/jobs/"+p, "", nil); rec.Code != 404 {
			t.Errorf("delete %q = %d, want 404", p, rec.Code)
		}
		if rec := do(s, "GET", "/v1/jobs/"+p+"/events", "", nil); rec.Code != 404 {
			t.Errorf("events %q = %d, want 404", p, rec.Code)
		}
	}
}

func TestDeleteFinishedJobRemovesItsFiles(t *testing.T) {
	dir := t.TempDir()
	s, fl := newExecServer(t, dir, func(c *Config) { c.MaxJobs = 1 })
	fl.add(script{code: 0}, script{code: 0})
	startExec(t, s, fl)
	id := submit(t, s)
	waitStatus(t, s, id, StatusSucceeded)
	if rec := do(s, "POST", "/v1/jobs", validSpec, nil); rec.Code != 429 {
		t.Fatalf("store at its cap accepted a job: %d", rec.Code)
	}
	if rec := do(s, "DELETE", "/v1/jobs/"+id, "", nil); rec.Code != 204 {
		t.Fatalf("delete finished = %d, want 204", rec.Code)
	}
	if _, err := os.Stat(filepath.Join(dir, "jobs", id)); !os.IsNotExist(err) {
		t.Fatalf("job directory still exists after delete: %v", err)
	}
	if rec := do(s, "GET", "/v1/jobs/"+id, "", nil); rec.Code != 404 {
		t.Fatalf("deleted job still readable: %d", rec.Code)
	}
	id2 := submit(t, s) // the freed slot can be used again
	waitStatus(t, s, id2, StatusSucceeded)
}

func TestDeleteActiveJobIsRefused(t *testing.T) {
	s, fl := newExecServer(t, t.TempDir(), nil)
	fl.add(script{hold: newGate()})
	startExec(t, s, fl)
	id := submit(t, s)
	waitStatus(t, s, id, StatusRunning)
	if rec := do(s, "DELETE", "/v1/jobs/"+id, "", nil); rec.Code != 409 {
		t.Fatalf("delete of a running job = %d, want 409", rec.Code)
	}
}

// ---------------------------------------------------------------- path safety

func TestPathSafety(t *testing.T) {
	dir := t.TempDir()
	s, fl := newExecServer(t, dir, nil)
	fl.add(script{code: 0})
	startExec(t, s, fl)
	// A hostile-looking name is rejected by the schema, and a legal one never reaches the filesystem.
	if rec := do(s, "POST", "/v1/jobs", strings.Replace(validSpec, "name: t1", "name: ../../evil", 1), nil); rec.Code != 400 {
		t.Fatalf("path-like metadata.name = %d, want 400", rec.Code)
	}
	id := submit(t, s)
	waitStatus(t, s, id, StatusSucceeded)
	ents, _ := os.ReadDir(filepath.Join(dir, "jobs"))
	if len(ents) != 1 || ents[0].Name() != id || !idPattern.MatchString(ents[0].Name()) {
		t.Fatalf("job directories = %v, want exactly %s", ents, id)
	}
	for _, name := range []string{"../x", "job-1", "JOB-12345678", "job-123456789", "job-1234567g", ""} {
		if validID(name) {
			t.Errorf("validID(%q) = true", name)
		}
	}
	defer func() {
		if recover() == nil {
			t.Fatal("jobDir accepted an invalid id")
		}
	}()
	s.jobs.jobDir("../escape")
}

// ---------------------------------------------------------------- events

func TestEventsReplayLiveFollowAndRestart(t *testing.T) {
	dir := t.TempDir()
	s, fl := newExecServer(t, dir, nil)
	release := newGate()
	fl.add(script{lines: []string{evStarted, evStep}, hold: release, code: 0})
	startExec(t, s, fl)
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(ts.Close)

	id := submit(t, s)
	waitStatus(t, s, id, StatusRunning)
	waitFor(t, "the worker's first events", func() bool { return len(eventsOf(t, s, id)) >= 3 })

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", ts.URL+"/v1/jobs/"+id+"/events", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "application/x-ndjson" {
		t.Fatalf("content type = %q", ct)
	}
	sc := bufio.NewScanner(resp.Body)
	var seen []string
	next := func() map[string]any {
		if !sc.Scan() {
			t.Fatalf("stream ended early (saw %v): %v", seen, sc.Err())
		}
		var m map[string]any
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("not JSON: %q", sc.Bytes())
		}
		seen = append(seen, fmt.Sprint(m["event"]))
		return m
	}
	for _, want := range []string{"job_started", "started", "step"} { // replayed history
		if got := next()["event"]; got != want {
			t.Fatalf("replay event = %v, want %s", got, want)
		}
	}
	release.open() // the job ends: the live stream must deliver the end and then close
	for _, want := range []string{"job_end"} {
		if got := next()["event"]; got != want {
			t.Fatalf("live event = %v, want %s", got, want)
		}
	}
	if sc.Scan() {
		t.Fatalf("stream stayed open after the job ended: %q", sc.Text())
	}

	// A finished job is served from the file, including after a restart.
	waitStatus(t, s, id, StatusSucceeded)
	s2, _ := newExecServer(t, dir, func(c *Config) { c.Execute = false })
	if got := strings.Join(kinds(eventsOf(t, s2, id)), ","); got != "job_started,started,step,job_end" {
		t.Fatalf("events after restart = %s", got)
	}
	tail := do(s2, "GET", "/v1/jobs/"+id+"/events?follow=0&tail=2", "", nil).Body.String()
	if n := len(strings.Split(strings.TrimSpace(tail), "\n")); n != 2 {
		t.Fatalf("tail=2 returned %d lines: %q", n, tail)
	}
	for _, bad := range []string{"tail=0", "tail=1001", "tail=abc"} {
		if rec := do(s2, "GET", "/v1/jobs/"+id+"/events?"+bad, "", nil); rec.Code != 400 {
			t.Errorf("?%s = %d, want 400", bad, rec.Code)
		}
	}
}

func TestEventStreamEndsWhenTheAgentShutsDown(t *testing.T) {
	s, fl := newExecServer(t, t.TempDir(), nil)
	fl.add(script{lines: []string{evStarted}, hold: newGate()})
	ctx, cancel := context.WithCancel(context.Background())
	stopExec := s.StartExecutor(ctx)
	t.Cleanup(func() { fl.releaseAll(); stopExec(); cancel() })
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(ts.Close)
	id := submit(t, s)
	waitStatus(t, s, id, StatusRunning)

	resp, err := http.Get(ts.URL + "/v1/jobs/" + id + "/events")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	done := make(chan struct{})
	go func() { _, _ = io.Copy(io.Discard, resp.Body); close(done) }()
	s.closeOnce.Do(func() { close(s.closing) })
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("event stream did not end when the agent started shutting down")
	}
}

// ---------------------------------------------------------------- shutdown

func TestShutdownRequeuesAPreemptedJobWithoutUsingARetry(t *testing.T) {
	dir := t.TempDir()
	s, fl := newExecServer(t, dir, func(c *Config) { c.MaxRetries = 0 }) // zero retries: only the shutdown rule can requeue
	fl.add(script{hold: newGate(), onTerm: ip(75)})
	ctx, cancel := context.WithCancel(context.Background())
	stop := s.StartExecutor(ctx)
	id := submit(t, s)
	waitStatus(t, s, id, StatusRunning)

	cancel()
	stop() // what Serve does on SIGTERM: stop dispatching, ask workers to checkpoint, wait for them
	j := jobOf(t, s, id)
	if j.Status != StatusQueued || j.Attempts != 0 || fl.children[0].signals() != 1 {
		t.Fatalf("after shutdown job = %+v signals=%d, want queued with the attempt refunded", j, fl.children[0].signals())
	}
	// The next agent resumes it.
	s2, fl2 := newExecServer(t, dir, nil)
	fl2.add(script{lines: []string{evFinished}, code: 0})
	startExec(t, s2, fl2)
	if j2 := waitStatus(t, s2, id, StatusSucceeded); j2.Attempts != 1 {
		t.Fatalf("resumed job = %+v, want 1 attempt", j2)
	}
}

func TestShutdownMarksANonCheckpointingJobInterrupted(t *testing.T) {
	s, fl := newExecServer(t, t.TempDir(), nil)
	fl.add(script{hold: newGate()}) // ignores SIGTERM: killed after the grace period, exit -> crash code
	ctx, cancel := context.WithCancel(context.Background())
	stop := s.StartExecutor(ctx)
	id := submit(t, s)
	waitStatus(t, s, id, StatusRunning)
	cancel()
	stop()
	if j := jobOf(t, s, id); j.Status != StatusInterrupted {
		t.Fatalf("job = %+v, want interrupted", j)
	}
}

// ---------------------------------------------------------------- metrics

func TestMetricsJobGaugesAndDurationHistogram(t *testing.T) {
	s, fl := newExecServer(t, t.TempDir(), nil)
	fl.add(script{code: 0}, script{code: 3}, script{hold: newGate()})
	startExec(t, s, fl)
	a, b, c := submit(t, s), submit(t, s), submit(t, s)
	waitStatus(t, s, a, StatusSucceeded)
	waitStatus(t, s, b, StatusFailed)
	waitStatus(t, s, c, StatusRunning)
	d := submit(t, s) // stays queued behind c

	m := parseMetrics(t, do(s, "GET", "/metrics", "", nil).Body.String())
	for series, want := range map[string]float64{
		`forgectl_agent_jobs{status="queued"}`:                                      1,
		`forgectl_agent_jobs{status="running"}`:                                     1,
		`forgectl_agent_jobs{status="succeeded"}`:                                   1,
		`forgectl_agent_jobs{status="failed"}`:                                      1,
		`forgectl_agent_jobs{status="cancelled"}`:                                   0,
		`forgectl_agent_jobs{status="interrupted"}`:                                 0,
		"forgectl_agent_jobs_queued":                                                1,
		`forgectl_agent_job_duration_seconds_count{status="succeeded"}`:             1,
		`forgectl_agent_job_duration_seconds_count{status="failed"}`:                1,
		`forgectl_agent_job_duration_seconds_count{status="cancelled"}`:             0,
		`forgectl_agent_job_duration_seconds_count{status="interrupted"}`:           0,
		`forgectl_agent_job_duration_seconds_bucket{status="succeeded",le="30"}`:    1, // real clock: a ms-long job
		`forgectl_agent_job_duration_seconds_bucket{status="succeeded",le="43200"}`: 1, // buckets are cumulative
		`forgectl_agent_job_duration_seconds_bucket{status="succeeded",le="+Inf"}`:  1,
		`forgectl_agent_job_duration_seconds_bucket{status="failed",le="30"}`:       1,
		`forgectl_agent_job_duration_seconds_bucket{status="failed",le="+Inf"}`:     1,
		`forgectl_agent_job_duration_seconds_bucket{status="cancelled",le="+Inf"}`:  0,
		`forgectl_agent_job_duration_seconds_sum{status="cancelled"}`:               0,
	} {
		if got, ok := m[series]; !ok || got != want {
			t.Errorf("%s = %v (present %v), want %v", series, got, ok, want)
		}
	}
	// Cardinality is bounded: job ids and names never appear in labels.
	for series := range m {
		if strings.Contains(series, a) || strings.Contains(series, d) || strings.Contains(series, "t1") {
			t.Errorf("series %q carries a per-job label", series)
		}
	}
}

// ---------------------------------------------------------------- event log units

func TestEventLogBoundsAndTruncation(t *testing.T) {
	l := openEventLog("")
	l.append([]byte(strings.Repeat("x", maxEventLine+1)))
	lines, _ := l.tail(1)
	if !strings.Contains(string(lines[0]), `"event":"truncated"`) || len(lines[0]) > 200 {
		t.Fatalf("oversized line was kept: %d bytes", len(lines[0]))
	}
	for i := 0; i < ringMaxLines+50; i++ {
		l.append([]byte(fmt.Sprintf(`{"event":"step","step":%d}`, i)))
	}
	ls, next := l.tail(ringMaxLines + 100)
	if len(ls) > ringMaxLines || next != uint64(ringMaxLines+51) {
		t.Fatalf("ring holds %d lines (next %d), want at most %d", len(ls), next, ringMaxLines)
	}
	if _, _, gap, _, _ := l.read(0); !gap {
		t.Fatal("a reader that fell behind the ring must be told about the gap")
	}
	l.append([]byte("")) // empty lines are dropped
	l.finish()
	l.append([]byte(`{"event":"late"}`)) // nothing after finish
	if _, n := l.tail(1); n != next {
		t.Fatal("append after finish changed the log")
	}
}

func TestReadTailFileDropsAPartialFirstLine(t *testing.T) {
	p := filepath.Join(t.TempDir(), "e.ndjson")
	var b strings.Builder
	for i := 0; i < 6; i++ {
		fmt.Fprintf(&b, `{"event":"step","step":%d,"pad":"%s"}`+"\n", i, strings.Repeat("p", 1<<20))
	}
	_ = os.WriteFile(p, []byte(b.String()), 0o600) // ~6 MiB > tailReadBytes: the first read line is cut in half
	lines := readTailFile(p, 100)
	for _, l := range lines {
		var m map[string]any
		if err := json.Unmarshal(l, &m); err != nil {
			t.Fatalf("tail returned a broken line: %.60q", l)
		}
	}
	if len(lines) == 0 || !strings.Contains(string(lines[len(lines)-1]), `"step":5`) {
		t.Fatalf("tail does not end at the last event")
	}
	if got := readTailFile(filepath.Join(t.TempDir(), "missing"), 5); got != nil {
		t.Fatalf("missing file = %v", got)
	}
}

func TestLineWriterSplitsAndFlushes(t *testing.T) {
	l := openEventLog("")
	w := &lineWriter{log: l}
	_, _ = w.Write([]byte(`{"a":1}` + "\n" + `{"b":2}` + "\n" + `{"c"`))
	_, _ = w.Write([]byte(`:3}`))
	w.flush()
	ls, _ := l.tail(10)
	var got []string
	for _, x := range ls {
		got = append(got, string(x))
	}
	if strings.Join(got, "|") != `{"a":1}|{"b":2}|{"c":3}` {
		t.Fatalf("lines = %v", got)
	}
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}
