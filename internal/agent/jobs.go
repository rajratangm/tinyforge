package agent

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"sync"
	"syscall"
	"time"

	"tinyforge.dev/forgectl/internal/jobspec"
)

// Job statuses. queued and running are active; the rest are terminal.
const (
	StatusQueued      = "queued"
	StatusRunning     = "running"
	StatusSucceeded   = "succeeded"
	StatusFailed      = "failed"
	StatusCancelled   = "cancelled"
	StatusInterrupted = "interrupted"
)

var (
	allStatuses      = []string{StatusQueued, StatusRunning, StatusSucceeded, StatusFailed, StatusCancelled, StatusInterrupted}
	terminalStatuses = []string{StatusSucceeded, StatusFailed, StatusCancelled, StatusInterrupted}
)

func isTerminal(status string) bool {
	switch status {
	case StatusSucceeded, StatusFailed, StatusCancelled, StatusInterrupted:
		return true
	}
	return false
}

var (
	errFull        = errors.New("job store is full")
	errNotFound    = errors.New("no such job")
	errNotTerminal = errors.New("job is not finished")
	errFinished    = errors.New("job already finished")
)

// Job is the public view of a TrainingJob the agent accepted. Pointers are only ever replaced, never mutated in
// place, so a copy of a Job is safe to read without the store lock.
type Job struct {
	Seq       int64      `json:"seq"` // submission order: the queue is oldest-first by this, never by clock time
	ID        string     `json:"id"`
	Name      string     `json:"name"`
	Namespace string     `json:"namespace"`
	Status    string     `json:"status"`
	Submitted time.Time  `json:"submitted"`
	Started   *time.Time `json:"started,omitempty"`  // start of the latest attempt
	Finished  *time.Time `json:"finished,omitempty"` // set only for terminal jobs
	ExitCode  *int       `json:"exit_code,omitempty"`
	Attempts  int        `json:"attempts"` // worker launches so far (retries after exit 75 add to it)
	Message   string     `json:"message,omitempty"`
}

// entry is a Job plus the in-process runtime state that is never persisted.
type entry struct {
	job       Job
	events    *eventLog      // non-nil while the job is active in this process (queued or running)
	sig       chan os.Signal // non-nil while running: stop requests for the runner
	cancelReq bool
	shutdown  bool // the agent itself is stopping, not the user
}

// jobStore keeps jobs in memory and, when dir is set, mirrors each one to <dir>/jobs/<id>/:
//
//	job.json      the Job record (atomic write)
//	spec.yaml     the spec exactly as submitted: this file is what the worker is given
//	events.ndjson the worker's JSON-lines events, verbatim, plus agent lifecycle events
//	worker.log    the worker's stderr (capped)
//	out/          the worker's --out directory
//
// Every path is derived from the generated job id. Nothing the client controls (metadata.name, labels, paths in
// the spec) ever reaches the filesystem layout.
type jobStore struct {
	mu    sync.Mutex
	max   int
	dir   string // "" = memory only (nothing persists, nothing can run)
	now   func() time.Time
	log   *log.Logger
	seq   int64 // last sequence number handed out
	order []string
	byID  map[string]*entry
}

var idPattern = regexp.MustCompile(`^job-[0-9a-f]{8}$`)

func validID(id string) bool { return idPattern.MatchString(id) }

func newJobStore(max int, dir string, now func() time.Time, lg *log.Logger) (*jobStore, error) {
	s := &jobStore{max: max, dir: dir, now: now, log: lg, byID: map[string]*entry{}}
	if dir == "" {
		return s, nil
	}
	if err := os.MkdirAll(filepath.Join(dir, "jobs"), 0o700); err != nil {
		return nil, fmt.Errorf("state dir: %w", err)
	}
	if err := s.load(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *jobStore) jobDir(id string) string {
	if !validID(id) {
		panic("agent: invalid job id used as a path: " + id) // programming error: ids are generated or validated
	}
	return filepath.Join(s.dir, "jobs", id)
}

func (s *jobStore) specPath(id string) string   { return filepath.Join(s.jobDir(id), "spec.yaml") }
func (s *jobStore) outDir(id string) string     { return filepath.Join(s.jobDir(id), "out") }
func (s *jobStore) eventsPath(id string) string { return s.pathIn(id, "events.ndjson") }
func (s *jobStore) logPath(id string) string    { return s.pathIn(id, "worker.log") }

func (s *jobStore) pathIn(id, name string) string {
	if s.dir == "" {
		return ""
	}
	return filepath.Join(s.jobDir(id), name)
}

// load restores jobs after a restart. A job that was running is marked interrupted and is NOT re-run: the agent
// cannot know how far it got, and re-running could repeat side effects. Queued jobs stay queued.
func (s *jobStore) load() error {
	ents, err := os.ReadDir(filepath.Join(s.dir, "jobs"))
	if err != nil {
		return fmt.Errorf("state dir: %w", err)
	}
	var loaded []*entry
	for _, d := range ents {
		if !d.IsDir() || !validID(d.Name()) {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(s.dir, "jobs", d.Name(), "job.json"))
		if err != nil {
			s.log.Printf("skipping a job directory without a readable record")
			continue
		}
		var j Job
		if err := json.Unmarshal(raw, &j); err != nil || j.ID != d.Name() {
			s.log.Printf("skipping a job directory with a corrupt record")
			continue
		}
		e := &entry{job: j}
		switch {
		case j.Status == StatusRunning:
			t := s.now()
			e.job.Status, e.job.Finished = StatusInterrupted, &t
			e.job.Message = "the agent stopped while this job was running; it was not re-run automatically. " +
				"Resubmit the spec to resume from the last checkpoint."
			if err := s.persist(e); err != nil {
				return err
			}
		case j.Status == StatusQueued:
			e.events = openEventLog(s.eventsPath(j.ID))
		case !isTerminal(j.Status):
			s.log.Printf("skipping a job with unknown status")
			continue
		}
		loaded = append(loaded, e)
	}
	sort.Slice(loaded, func(i, k int) bool {
		a, b := loaded[i].job, loaded[k].job
		if a.Seq != b.Seq {
			return a.Seq < b.Seq
		}
		if !a.Submitted.Equal(b.Submitted) {
			return a.Submitted.Before(b.Submitted)
		}
		return a.ID < b.ID
	})
	for _, e := range loaded {
		s.byID[e.job.ID] = e
		s.order = append(s.order, e.job.ID)
		if e.job.Seq > s.seq {
			s.seq = e.job.Seq
		}
	}
	return nil
}

// persist writes job.json atomically. Caller holds s.mu (or owns e exclusively).
func (s *jobStore) persist(e *entry) error {
	if s.dir == "" {
		return nil
	}
	b, err := json.MarshalIndent(e.job, "", "  ")
	if err != nil {
		return err
	}
	return writeFileAtomic(filepath.Join(s.jobDir(e.job.ID), "job.json"), b, 0o600)
}

func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, perm); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// add stores a new queued job. raw is the spec exactly as submitted.
func (s *jobStore) add(j *jobspec.Job, raw []byte) (Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.order) >= s.max {
		return Job{}, errFull
	}
	id := newID()
	for _, taken := s.byID[id]; taken; _, taken = s.byID[id] {
		id = newID()
	}
	e := &entry{job: Job{Seq: s.seq + 1, ID: id, Name: j.Metadata.Name, Namespace: j.Metadata.Namespace,
		Status: StatusQueued, Submitted: s.now()}}
	if s.dir != "" {
		if err := os.MkdirAll(s.outDir(id), 0o700); err != nil {
			return Job{}, err
		}
		if err := os.WriteFile(s.specPath(id), raw, 0o600); err != nil {
			_ = os.RemoveAll(s.jobDir(id))
			return Job{}, err
		}
		if err := s.persist(e); err != nil {
			_ = os.RemoveAll(s.jobDir(id))
			return Job{}, err
		}
	}
	e.events = openEventLog(s.eventsPath(id))
	s.seq++
	s.byID[id] = e
	s.order = append(s.order, id)
	return e.job, nil
}

func (s *jobStore) get(id string) (Job, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.byID[id]
	if !ok {
		return Job{}, false
	}
	return e.job, true
}

// getWithEvents returns the job and its live event log (nil once the job is terminal).
func (s *jobStore) getWithEvents(id string) (Job, *eventLog, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.byID[id]
	if !ok {
		return Job{}, nil, false
	}
	return e.job, e.events, true
}

func (s *jobStore) list() []Job {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Job, 0, len(s.order))
	for _, id := range s.order {
		out = append(out, s.byID[id].job)
	}
	return out
}

func (s *jobStore) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.order)
}

func (s *jobStore) counts() map[string]int {
	s.mu.Lock()
	defer s.mu.Unlock()
	m := make(map[string]int, len(allStatuses))
	for _, st := range allStatuses {
		m[st] = 0
	}
	for _, e := range s.byID {
		m[e.job.Status]++
	}
	return m
}

// claim is what the executor needs to run one attempt.
type claim struct {
	job    Job
	sig    chan os.Signal
	events *eventLog
}

// claimNext marks the oldest queued job running, unless maxConcurrent jobs already run. It returns nil if there is
// nothing to start.
func (s *jobStore) claimNext(maxConcurrent int) (*claim, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	running := 0
	for _, e := range s.byID {
		if e.job.Status == StatusRunning {
			running++
		}
	}
	if running >= maxConcurrent {
		return nil, nil
	}
	for _, id := range s.order {
		e := s.byID[id]
		if e.job.Status != StatusQueued {
			continue
		}
		t := s.now()
		e.job.Status, e.job.Started, e.job.Finished, e.job.Message = StatusRunning, &t, nil, ""
		e.job.Attempts++
		e.sig, e.cancelReq, e.shutdown = make(chan os.Signal, 2), false, false
		if e.events == nil {
			e.events = openEventLog(s.eventsPath(id))
		}
		if err := s.persist(e); err != nil {
			e.job.Status = StatusFailed
			e.job.Message = "could not persist job state: " + err.Error()
			e.job.Attempts--
			return nil, err
		}
		return &claim{job: e.job, sig: e.sig, events: e.events}, nil
	}
	return nil, nil
}

// finish records the outcome of an attempt. requeue puts the job back in the queue (preempted, will resume).
func (s *jobStore) finish(id, status string, code *int, message string) (Job, *eventLog) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e := s.byID[id]
	if e == nil {
		return Job{}, nil
	}
	e.job.Status, e.job.ExitCode, e.job.Message = status, code, message
	e.sig = nil
	var ev *eventLog
	if isTerminal(status) {
		t := s.now()
		e.job.Finished = &t
		ev, e.events = e.events, nil
	} else {
		e.job.Finished = nil
		ev = e.events
	}
	if err := s.persist(e); err != nil {
		s.log.Printf("could not persist job state: %v", err)
	}
	return e.job, ev
}

// refundAttempt un-counts the latest attempt (an attempt cut short by the agent stopping is not the job's fault).
func (s *jobStore) refundAttempt(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.byID[id]; e != nil && e.job.Attempts > 0 {
		e.job.Attempts--
		if err := s.persist(e); err != nil {
			s.log.Printf("could not persist job state: %v", err)
		}
	}
}

// cancel stops a job. A queued job is cancelled at once. A running job is asked to stop gracefully (the runner
// forwards SIGTERM, then kills after its grace period); force=true on an already-cancelling job sends a second
// request, which makes the runner kill the worker immediately.
func (s *jobStore) cancel(id string, force bool) (Job, *eventLog, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e := s.byID[id]
	if e == nil {
		return Job{}, nil, errNotFound
	}
	switch e.job.Status {
	case StatusQueued:
		t := s.now()
		e.job.Status, e.job.Finished, e.job.Message = StatusCancelled, &t, "cancelled before it started"
		ev := e.events
		e.events = nil
		if err := s.persist(e); err != nil {
			s.log.Printf("could not persist job state: %v", err)
		}
		return e.job, ev, nil
	case StatusRunning:
		if !e.cancelReq || force {
			e.cancelReq = true
			select {
			case e.sig <- syscall.SIGTERM:
			default:
			}
		}
		return e.job, nil, nil
	default:
		return e.job, nil, errFinished
	}
}

// signalAllRunning asks every running job to stop because the agent is shutting down.
func (s *jobStore) signalAllRunning() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, e := range s.byID {
		if e.job.Status == StatusRunning {
			e.shutdown = true
			select {
			case e.sig <- syscall.SIGTERM:
			default:
			}
		}
	}
}

// flags returns the stop reasons recorded for a running job.
func (s *jobStore) flags(id string) (cancelReq, shutdown bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.byID[id]; e != nil {
		return e.cancelReq, e.shutdown
	}
	return false, false
}

// remove deletes a finished job and its directory.
func (s *jobStore) remove(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	e := s.byID[id]
	if e == nil {
		return errNotFound
	}
	if !isTerminal(e.job.Status) {
		return errNotTerminal
	}
	if s.dir != "" {
		if err := os.RemoveAll(s.jobDir(id)); err != nil {
			return err
		}
	}
	delete(s.byID, id)
	for i, v := range s.order {
		if v == id {
			s.order = append(s.order[:i], s.order[i+1:]...)
			break
		}
	}
	return nil
}

func newID() string {
	b := make([]byte, 4)
	if _, err := rand.Read(b); err != nil {
		panic("agent: crypto/rand failed: " + err.Error())
	}
	return "job-" + hex.EncodeToString(b)
}
