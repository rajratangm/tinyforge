package agent

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"tinyforge.dev/forgectl/internal/runner"
)

const (
	defaultMaxConcurrent = 1
	defaultCancelGrace   = 30 * time.Second
	workerLogCap         = 1 << 20 // worker stderr kept on disk per job
	pollEvery            = 2 * time.Second
)

// StartExecutor starts the scheduler: it runs queued jobs, at most MaxConcurrent at a time, oldest first. The
// returned stop function stops dispatching, asks running jobs to checkpoint and exit (SIGTERM, then a kill after
// CancelGrace), and waits for them. A job that exits 75 in response goes back to the queue, so it resumes from its
// checkpoint when the agent starts again; any other exit marks it interrupted.
func (s *Server) StartExecutor(ctx context.Context) (stop func()) {
	ctx, cancel := context.WithCancel(ctx)
	loopDone := make(chan struct{})
	go func() {
		defer close(loopDone)
		s.schedule(ctx)
	}()
	var once sync.Once
	return func() {
		once.Do(func() {
			cancel()
			<-loopDone
			s.jobs.signalAllRunning()
			drained := make(chan struct{})
			go func() { s.wg.Wait(); close(drained) }()
			select {
			case <-drained:
			case <-time.After(s.cfg.CancelGrace + 10*time.Second):
				s.cfg.Log.Printf("some jobs did not exit in time at shutdown")
			}
		})
	}
}

func (s *Server) kick() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

func (s *Server) schedule(ctx context.Context) {
	for {
		if ctx.Err() != nil {
			return
		}
		s.dispatch()
		select {
		case <-ctx.Done():
			return
		case <-s.wake:
		case <-time.After(pollEvery):
		}
	}
}

func (s *Server) dispatch() {
	for {
		c, err := s.jobs.claimNext(s.cfg.MaxConcurrent)
		if err != nil {
			s.cfg.Log.Printf("could not start a job: %v", err)
			return
		}
		if c == nil {
			return
		}
		s.wg.Add(1)
		go s.execute(c)
	}
}

// execute runs one attempt of a claimed job and records the outcome.
func (s *Server) execute(c *claim) {
	defer s.wg.Done()
	defer s.kick() // a slot is free (or the job was re-queued): look at the queue again
	id := c.job.ID

	c.events.appendEvent("job_started", map[string]any{"attempt": c.job.Attempts})

	// Fail closed and loudly if there is no worker to run, instead of letting the runner report an opaque crash.
	if _, _, err := runner.FindPython(s.cfg.Python, s.cfg.Getenv, s.cfg.Cwd, s.cfg.Exists, s.cfg.LookPath); err != nil {
		c.events.appendEvent("diagnostic", map[string]any{"code": "AG001", "level": "error",
			"message": "no Python worker interpreter found: " + err.Error(),
			"fix":     "Start the agent with --python, set FORGECTL_PYTHON, or install tinyforge in a .venv next to the agent."})
		s.conclude(c, decision{status: StatusFailed, code: intp(runner.ExitCrash),
			message: "AG001: no Python worker found (--python, FORGECTL_PYTHON, .venv or PATH); the job was not started"})
		return
	}

	var logFile *cappedFile
	if p := s.jobs.logPath(id); p != "" {
		if f, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600); err == nil {
			logFile = &cappedFile{f: f, left: workerLogCap}
			defer f.Close()
		}
	}
	out := &lineWriter{log: c.events}
	opts := runner.Options{
		SpecPath: s.jobs.specPath(id),
		Out:      s.jobs.outDir(id),
		Python:   s.cfg.Python,
		JSON:     true, // raw worker lines on stdout; the human summary goes to the worker log
		Stdout:   out,
		Signals:  c.sig,
		Grace:    s.cfg.CancelGrace,
		Launch:   s.cfg.Launch,
		Now:      s.cfg.Now,
		Getenv:   s.cfg.Getenv,
		Cwd:      s.cfg.Cwd,
		Exists:   s.cfg.Exists,
		LookPath: s.cfg.LookPath,
	}
	if logFile != nil {
		opts.Stderr = logFile
	}
	code := runner.Run(opts)
	out.flush()

	cancelReq, shutdown := s.jobs.flags(id)
	s.conclude(c, decide(code, cancelReq, shutdown, c.job.Attempts, s.cfg.MaxRetries))
}

// decision is the outcome of one attempt.
type decision struct {
	status  string
	code    *int
	message string
}

// decide maps a worker exit code to a job outcome (spec/worker-contract.md):
//
//	0   succeeded
//	75  preempted: re-queued (the worker resumes from its checkpoint) up to maxRetries times, then failed
//	2,3,4,5 and anything else: failed, never retried (a retry would fail the same way)
//
// A stop requested by the user makes the job cancelled; a stop caused by the agent shutting down re-queues a
// preempted job without using up a retry and marks anything else interrupted. A job that finished successfully
// anyway is succeeded whatever was requested.
func decide(code int, cancelReq, shutdown bool, attempts, maxRetries int) decision {
	d := decision{code: intp(code)}
	switch {
	case code == runner.ExitOK:
		d.status = StatusSucceeded
		d.message = runner.Describe(code, false)
	case shutdown && code == runner.ExitPreempted:
		d.status, d.message = StatusQueued, "the agent is stopping: the worker checkpointed and the job will resume when the agent starts"
	case shutdown:
		d.status, d.message = StatusInterrupted, "the agent stopped while this job was running; resubmit the spec to resume from the last checkpoint"
	case cancelReq:
		d.status, d.message = StatusCancelled, "cancelled by request"
	case code == runner.ExitPreempted && attempts <= maxRetries:
		d.status = StatusQueued
		d.message = fmt.Sprintf("preempted (exit 75): retry %d of %d queued, the worker resumes from its checkpoint", attempts, maxRetries)
	case code == runner.ExitPreempted:
		d.status = StatusFailed
		d.message = fmt.Sprintf("preempted (exit 75) %d times: giving up after %d retries", attempts, maxRetries)
	default:
		d.status, d.message = StatusFailed, runner.Describe(code, false)
	}
	return d
}

func (s *Server) conclude(c *claim, d decision) {
	_, shutdown := s.jobs.flags(c.job.ID)
	job, ev := s.jobs.finish(c.job.ID, d.status, d.code, d.message)
	if job.ID == "" {
		return
	}
	switch {
	case d.status == StatusQueued:
		if shutdown {
			// Not a retry: the attempt did not count against the budget.
			s.jobs.refundAttempt(c.job.ID)
		}
		ev.appendEvent("job_requeued", map[string]any{"exit_code": derefInt(d.code), "message": d.message})
	default:
		ev.appendEvent("job_end", map[string]any{"status": d.status, "exit_code": derefInt(d.code), "message": d.message})
		ev.finish()
		if job.Started != nil && job.Finished != nil {
			s.durations.observe(job.Status, job.Finished.Sub(*job.Started).Seconds())
		}
	}
}

func intp(v int) *int { return &v }

func derefInt(p *int) any {
	if p == nil {
		return nil
	}
	return *p
}

// cappedFile writes to a file until the cap is reached and silently drops the rest, so a chatty worker cannot fill
// the disk through its stderr.
type cappedFile struct {
	f    *os.File
	left int64
}

func (c *cappedFile) Write(p []byte) (int, error) {
	n := len(p)
	if c.left <= 0 {
		return n, nil
	}
	if int64(len(p)) > c.left {
		p = p[:c.left]
	}
	w, _ := c.f.Write(p)
	c.left -= int64(w)
	return n, nil
}

// histogram is a fixed-bucket Prometheus histogram of job durations, one per terminal status (bounded labels).
var durationBuckets = []float64{30, 60, 300, 900, 1800, 3600, 7200, 14400, 43200}

type histogram struct {
	counts []uint64 // non-cumulative, one per bucket; the overflow goes to +Inf only
	sum    float64
	n      uint64
}

type durationHists struct {
	mu sync.Mutex
	by map[string]*histogram
}

func newDurationHists() *durationHists {
	d := &durationHists{by: map[string]*histogram{}}
	for _, st := range terminalStatuses {
		d.by[st] = &histogram{counts: make([]uint64, len(durationBuckets))}
	}
	return d
}

func (d *durationHists) observe(status string, seconds float64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	h := d.by[status]
	if h == nil || seconds < 0 {
		return
	}
	for i, le := range durationBuckets {
		if seconds <= le {
			h.counts[i]++
			break
		}
	}
	h.sum += seconds
	h.n++
}

// write renders the histogram family in exposition format (cumulative buckets).
func (d *durationHists) write(b *strings.Builder) {
	d.mu.Lock()
	defer d.mu.Unlock()
	const name = "forgectl_agent_job_duration_seconds"
	family(b, name, "histogram", "Duration of the latest attempt of finished jobs, by final status (resets when the agent restarts).")
	for _, st := range terminalStatuses {
		h := d.by[st]
		var cum uint64
		for i, le := range durationBuckets {
			cum += h.counts[i]
			sample(b, name+"_bucket", `status="`+st+`",le="`+strconv.FormatFloat(le, 'g', -1, 64)+`"`, float64(cum))
		}
		sample(b, name+"_bucket", `status="`+st+`",le="+Inf"`, float64(h.n))
		sample(b, name+"_sum", `status="`+st+`"`, h.sum)
		sample(b, name+"_count", `status="`+st+`"`, float64(h.n))
	}
}
