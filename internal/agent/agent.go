// Package agent is the forgectl node agent: a small HTTP service that reports this node's health (the doctor
// report, cached), accepts TrainingJob specs, and runs them one after another through the Python worker
// (`tinyforge worker run`, via internal/runner) with state persisted under a state directory. Transport: loopback
// by default; non-loopback needs TLS (optionally mutual TLS) plus a bearer token or client certificates, see
// CheckListen.
package agent

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"tinyforge.dev/forgectl/internal/doctor"
	"tinyforge.dev/forgectl/internal/jobspec"
	"tinyforge.dev/forgectl/internal/runner"
)

const (
	Version        = "0.1.0-dev"
	DefaultListen  = "127.0.0.1:7070"
	defaultMaxBody = 1 << 20 // 1 MiB
	defaultMaxJobs = 1000
)

// Config holds everything injectable for tests. Zero values get sensible defaults.
type Config struct {
	Listen       string // address the caller intends to listen on; checked by New
	Token        string // bearer token; empty = no auth (loopback only)
	Version      string
	Hostname     string
	Now          func() time.Time
	Doctor       func(ctx context.Context) *doctor.Report
	RefreshEvery time.Duration
	MaxBody      int64
	MaxJobs      int
	Log          *log.Logger

	// Transport security. TLSCertFile and TLSKeyFile enable TLS; ClientCAFile additionally requires and verifies
	// client certificates (mutual TLS). MTLSOnly drops the bearer-token requirement (client certificates are then
	// the only credential). InsecureHTTP permits plain HTTP on a non-loopback address (a token is still required).
	TLSCertFile, TLSKeyFile string
	ClientCAFile            string
	MTLSOnly                bool
	InsecureHTTP            bool
	CertReloadEvery         time.Duration // how often to re-read the key pair from disk; default 10 minutes

	// Job execution. StateDir persists jobs across restarts (empty = memory only, which cannot run anything).
	// Execute turns the scheduler on: Serve then runs queued jobs, at most MaxConcurrent at a time (default 1).
	// A job whose worker exits 75 (preempted) is re-queued up to MaxRetries times (0 = never). CancelGrace is how
	// long a stopping worker gets to checkpoint before it is killed (default 30 s). Everything below Python is for
	// tests; zero values mean the real thing (see runner.Options).
	StateDir      string
	Execute       bool
	MaxConcurrent int
	MaxRetries    int
	CancelGrace   time.Duration
	Python        string
	Launch        runner.Launcher
	Getenv        func(string) string
	Cwd           string
	Exists        func(string) bool
	LookPath      func(string) (string, error)
}

type Server struct {
	cfg       Config
	tokenHash [32]byte
	started   time.Time
	jobs      *jobStore
	tlsCfg    *tls.Config   // nil = plain HTTP
	certs     *certReloader // nil = plain HTTP

	wake      chan struct{}  // nudges the scheduler
	wg        sync.WaitGroup // running job attempts
	durations *durationHists
	closing   chan struct{} // closed when the agent starts shutting down: ends event streams
	closeOnce sync.Once

	mu         sync.RWMutex
	report     *doctor.Report
	reportedAt time.Time
}

func New(cfg Config) (*Server, error) {
	if cfg.Listen == "" {
		cfg.Listen = DefaultListen
	}
	tlsCfg, certs, err := buildTLSConfig(cfg)
	if err != nil {
		return nil, err
	}
	sec := Security{
		TLS: tlsCfg != nil, MTLS: cfg.ClientCAFile != "", InsecureHTTP: cfg.InsecureHTTP,
		Token: cfg.Token != "" && !cfg.MTLSOnly,
	}
	if err := CheckListen(cfg.Listen, sec); err != nil {
		return nil, err
	}
	if cfg.Version == "" {
		cfg.Version = Version
	}
	if cfg.Hostname == "" {
		cfg.Hostname, _ = os.Hostname()
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.Doctor == nil {
		cfg.Doctor = func(ctx context.Context) *doctor.Report { return doctor.Run(ctx, doctor.DefaultOptions()) }
	}
	if cfg.RefreshEvery <= 0 {
		cfg.RefreshEvery = 15 * time.Second
	}
	if cfg.MaxBody <= 0 {
		cfg.MaxBody = defaultMaxBody
	}
	if cfg.MaxJobs <= 0 {
		cfg.MaxJobs = defaultMaxJobs
	}
	if cfg.Log == nil {
		cfg.Log = log.New(os.Stderr, "agent: ", log.LstdFlags)
	}
	if cfg.CertReloadEvery <= 0 {
		cfg.CertReloadEvery = 10 * time.Minute
	}
	if cfg.MaxConcurrent <= 0 {
		cfg.MaxConcurrent = defaultMaxConcurrent
	}
	if cfg.MaxRetries < 0 {
		cfg.MaxRetries = 0
	}
	if cfg.CancelGrace <= 0 {
		cfg.CancelGrace = defaultCancelGrace
	}
	if cfg.Execute && cfg.StateDir == "" {
		return nil, errors.New("job execution needs a state directory (--state-dir): jobs, specs and events live there")
	}
	store, err := newJobStore(cfg.MaxJobs, cfg.StateDir, cfg.Now, cfg.Log)
	if err != nil {
		return nil, err
	}
	s := &Server{cfg: cfg, started: cfg.Now(), jobs: store, tlsCfg: tlsCfg, certs: certs,
		wake: make(chan struct{}, 1), durations: newDurationHists(), closing: make(chan struct{})}
	if cfg.Token != "" {
		s.tokenHash = sha256.Sum256([]byte(cfg.Token))
	}
	if tlsCfg == nil && !sec.Token && !isLoopbackListen(cfg.Listen) {
		// Unreachable via CheckListen, kept as a guard against future edits.
		return nil, errors.New("internal: non-loopback listener without TLS or token")
	}
	if tlsCfg == nil && cfg.InsecureHTTP && !isLoopbackListen(cfg.Listen) {
		cfg.Log.Printf("WARNING: serving PLAIN HTTP on a non-loopback address: the bearer token and job specs " +
			"cross the network unencrypted. Use --tls-cert/--tls-key instead.")
	}
	return s, nil
}

func isLoopbackListen(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	return err == nil && isLoopbackHost(host)
}

// Refresh runs the doctor once and caches the report. Serve calls it every RefreshEvery; requests never do.
func (s *Server) Refresh(ctx context.Context) {
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	rep := s.cfg.Doctor(ctx)
	s.mu.Lock()
	s.report, s.reportedAt = rep, s.cfg.Now()
	s.mu.Unlock()
}

func (s *Server) snapshot() (*doctor.Report, time.Time) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.report, s.reportedAt
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.healthz)
	mux.HandleFunc("GET /readyz", s.readyz)
	mux.HandleFunc("GET /v1/node", s.auth(s.node))
	mux.HandleFunc("GET /metrics", s.auth(s.metrics))
	mux.HandleFunc("POST /v1/jobs", s.auth(s.submitJob))
	mux.HandleFunc("GET /v1/jobs", s.auth(s.listJobs))
	mux.HandleFunc("GET /v1/jobs/{id}", s.auth(s.getJob))
	mux.HandleFunc("GET /v1/jobs/{id}/events", s.auth(s.jobEvents))
	mux.HandleFunc("POST /v1/jobs/{id}/cancel", s.auth(s.cancelJob))
	mux.HandleFunc("DELETE /v1/jobs/{id}", s.auth(s.deleteJob))
	return s.logRequests(mux)
}

// auth requires "Authorization: Bearer <token>" when a token is configured: 401 if absent, 403 if wrong.
// Tokens are compared as SHA-256 digests in constant time so neither value nor length leaks. With MTLSOnly the
// TLS layer has already verified a client certificate, so no token is checked.
func (s *Server) auth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.Token == "" || s.cfg.MTLSOnly {
			next(w, r)
			return
		}
		h := r.Header.Get("Authorization")
		const prefix = "Bearer "
		if len(h) < len(prefix) || !strings.EqualFold(h[:len(prefix)], prefix) {
			w.Header().Set("WWW-Authenticate", `Bearer realm="forgectl-agent"`)
			writeJSON(w, http.StatusUnauthorized, errBody("missing bearer token"))
			return
		}
		got := sha256.Sum256([]byte(h[len(prefix):]))
		if subtle.ConstantTimeCompare(got[:], s.tokenHash[:]) != 1 {
			writeJSON(w, http.StatusForbidden, errBody("invalid token"))
			return
		}
		next(w, r)
	}
}

type statusRecorder struct {
	http.ResponseWriter
	code int
}

func (r *statusRecorder) WriteHeader(c int) { r.code = c; r.ResponseWriter.WriteHeader(c) }

// Unwrap lets http.NewResponseController reach the real writer: without it the event stream could neither flush
// nor extend its write deadline, so it would sit in a buffer and die at the server's WriteTimeout.
func (r *statusRecorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

// logRequests logs method, path and status only: never headers (the token) or bodies (job specs).
func (s *Server) logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := &statusRecorder{ResponseWriter: w, code: http.StatusOK}
		next.ServeHTTP(rec, r)
		s.cfg.Log.Printf("%s %s -> %d", r.Method, r.URL.Path, rec.code)
	})
}

func (s *Server) healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "version": s.cfg.Version})
}

// readyz is ready once the first doctor report exists, so a probe never sees an empty /v1/node.
func (s *Server) readyz(w http.ResponseWriter, _ *http.Request) {
	if rep, _ := s.snapshot(); rep == nil {
		writeJSON(w, http.StatusServiceUnavailable, errBody("first doctor report not ready yet"))
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

type nodeInfo struct {
	Hostname         string         `json:"hostname"`
	OS               string         `json:"os"`
	Arch             string         `json:"arch"`
	AgentVersion     string         `json:"agent_version"`
	UptimeSeconds    float64        `json:"uptime_seconds"`
	ReportAgeSeconds float64        `json:"report_age_seconds"`
	Report           *doctor.Report `json:"report"`
}

func (s *Server) node(w http.ResponseWriter, _ *http.Request) {
	rep, at := s.snapshot()
	if rep == nil {
		writeJSON(w, http.StatusServiceUnavailable, errBody("first doctor report not ready yet"))
		return
	}
	now := s.cfg.Now()
	writeJSON(w, http.StatusOK, nodeInfo{
		Hostname: s.cfg.Hostname, OS: rep.System.OS, Arch: rep.System.Arch, AgentVersion: s.cfg.Version,
		UptimeSeconds: now.Sub(s.started).Seconds(), ReportAgeSeconds: now.Sub(at).Seconds(), Report: rep,
	})
}

func (s *Server) submitJob(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, s.cfg.MaxBody)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeJSON(w, http.StatusRequestEntityTooLarge, errBody(fmt.Sprintf("body exceeds %d bytes", s.cfg.MaxBody)))
			return
		}
		writeJSON(w, http.StatusBadRequest, errBody("could not read body"))
		return
	}
	j, err := jobspec.Parse(body)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, errBody(sanitizeSpecError(err.Error())))
		return
	}
	job, err := s.jobs.add(j, body)
	if errors.Is(err, errFull) {
		writeJSON(w, http.StatusTooManyRequests, errBody(fmt.Sprintf("job store is full (%d jobs): delete finished jobs", s.cfg.MaxJobs)))
		return
	}
	if err != nil { // never accept a job that would vanish on restart
		s.cfg.Log.Printf("could not persist a submitted job: %v", err)
		writeJSON(w, http.StatusInternalServerError, errBody("could not persist the job"))
		return
	}
	s.kick()
	writeJSON(w, http.StatusAccepted, job)
}

// cancelJob: a queued job is cancelled (200); a running job is asked to stop (202, final status follows once the
// worker exits; ?force=1 on a job that is already stopping kills the worker); a finished job is a 409.
func (s *Server) cancelJob(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	force := r.URL.Query().Get("force") == "1"
	if !validID(id) {
		writeJSON(w, http.StatusNotFound, errBody("no such job"))
		return
	}
	job, ev, err := s.jobs.cancel(id, force)
	switch {
	case errors.Is(err, errNotFound):
		writeJSON(w, http.StatusNotFound, errBody("no such job"))
	case errors.Is(err, errFinished):
		writeJSON(w, http.StatusConflict, errBody("job already finished ("+job.Status+")"))
	case err != nil:
		writeJSON(w, http.StatusInternalServerError, errBody("could not cancel the job"))
	case job.Status == StatusCancelled: // was queued: done now
		if ev != nil {
			ev.appendEvent("job_end", map[string]any{"status": job.Status, "message": job.Message})
			ev.finish()
		}
		writeJSON(w, http.StatusOK, job)
	default: // running: stopping
		writeJSON(w, http.StatusAccepted, job)
	}
}

// deleteJob removes a finished job and its files; an active job must be cancelled first (409).
func (s *Server) deleteJob(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !validID(id) {
		writeJSON(w, http.StatusNotFound, errBody("no such job"))
		return
	}
	switch err := s.jobs.remove(id); {
	case errors.Is(err, errNotFound):
		writeJSON(w, http.StatusNotFound, errBody("no such job"))
	case errors.Is(err, errNotTerminal):
		writeJSON(w, http.StatusConflict, errBody("job is still active: cancel it first"))
	case err != nil:
		s.cfg.Log.Printf("could not delete a job: %v", err)
		writeJSON(w, http.StatusInternalServerError, errBody("could not delete the job"))
	default:
		w.WriteHeader(http.StatusNoContent)
	}
}

func (s *Server) listJobs(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"jobs": s.jobs.list()})
}

func (s *Server) getJob(w http.ResponseWriter, r *http.Request) {
	job, ok := s.jobs.get(r.PathValue("id"))
	if !ok {
		writeJSON(w, http.StatusNotFound, errBody("no such job"))
		return
	}
	writeJSON(w, http.StatusOK, job)
}

func errBody(msg string) map[string]string { return map[string]string{"error": msg} }

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

// Serve runs the HTTP server on ln plus the doctor refresh loop until ctx is cancelled, then shuts down
// gracefully (in-flight requests get 30 s). It returns nil on a clean shutdown.
func (s *Server) Serve(ctx context.Context, ln net.Listener) error {
	srv := &http.Server{
		Handler:           s.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
		ErrorLog:          s.cfg.Log,
	}
	loopCtx, stopLoop := context.WithCancel(ctx)
	defer stopLoop()
	go s.refreshLoop(loopCtx)

	// Shutdown order: end event streams, stop the scheduler and let running workers checkpoint (SIGTERM, then a
	// kill after CancelGrace), then drain the HTTP server.
	stopExec := func() {}
	if s.cfg.Execute {
		stopExec = s.StartExecutor(ctx)
	}
	defer stopExec()
	srv.RegisterOnShutdown(func() { s.closeOnce.Do(func() { close(s.closing) }) })

	errc := make(chan error, 1)
	if s.tlsCfg != nil {
		srv.TLSConfig = s.tlsCfg
		go s.certReloadLoop(loopCtx)
		go func() { errc <- srv.ServeTLS(ln, "", "") }() // key pair comes from tlsCfg.GetCertificate
	} else {
		go func() { errc <- srv.Serve(ln) }()
	}
	select {
	case err := <-errc:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		s.closeOnce.Do(func() { close(s.closing) })
		stopExec()
		sctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := srv.Shutdown(sctx); err != nil {
			return err
		}
		<-errc
		return nil
	}
}

// certReloadLoop re-reads the key pair periodically so rotation needs no restart (SIGHUP triggers it sooner).
func (s *Server) certReloadLoop(ctx context.Context) {
	t := time.NewTicker(s.cfg.CertReloadEvery)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			_ = s.ReloadCerts() // failures are logged without paths and keep the previous certificate
		}
	}
}

func (s *Server) refreshLoop(ctx context.Context) {
	s.Refresh(ctx)
	t := time.NewTicker(s.cfg.RefreshEvery)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.Refresh(ctx)
		}
	}
}
