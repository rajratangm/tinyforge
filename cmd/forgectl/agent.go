package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"tinyforge.dev/forgectl/internal/agent"
)

// runAgent starts the node agent. The token comes from --token-file, else $FORGECTL_AGENT_TOKEN; it is never
// printed. Non-loopback listening needs TLS plus a token or client certificates (see agent.CheckListen).
func runAgent(args []string) int {
	fs := flag.NewFlagSet("agent", flag.ContinueOnError)
	listen := fs.String("listen", agent.DefaultListen, "address to listen on (non-loopback requires TLS and credentials)")
	tokenFile := fs.String("token-file", "", "file holding the bearer token (else $FORGECTL_AGENT_TOKEN)")
	tlsCert := fs.String("tls-cert", "", "PEM server certificate (enables TLS; re-read on SIGHUP and every 10 minutes)")
	tlsKey := fs.String("tls-key", "", "PEM private key for --tls-cert")
	clientCA := fs.String("tls-client-ca", "", "PEM CA bundle: require and verify client certificates (mutual TLS)")
	mtlsOnly := fs.Bool("mtls-only", false, "with --tls-client-ca: client certificates are the only credential (no bearer token)")
	insecure := fs.Bool("insecure-http", false, "allow PLAIN HTTP on a non-loopback address (a bearer token is still required)")
	stateDir := fs.String("state-dir", "agent-state", "directory for job state, specs, events and worker output (survives restarts)")
	maxConc := fs.Int("max-concurrent", 1, "jobs to run at once (1 per GPU is the safe choice)")
	maxRetries := fs.Int("max-retries", 2, "re-queue a job whose worker was preempted (exit 75) this many times; 0 disables")
	grace := fs.Duration("cancel-grace", 30*time.Second, "how long a stopping worker may take to checkpoint before it is killed")
	python := fs.String("python", "", "Python interpreter that runs the worker (else $FORGECTL_PYTHON, a project .venv, or PATH)")
	noExec := fs.Bool("no-exec", false, "validate and queue jobs but do not run them")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	absState, err := filepath.Abs(*stateDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "FAIL: bad --state-dir")
		return 2
	}
	token := os.Getenv("FORGECTL_AGENT_TOKEN")
	if *tokenFile != "" {
		b, err := os.ReadFile(*tokenFile)
		if err != nil {
			fmt.Fprintln(os.Stderr, "FAIL: cannot read token file")
			return 2
		}
		if token = strings.TrimSpace(string(b)); token == "" {
			fmt.Fprintln(os.Stderr, "FAIL: token file is empty")
			return 2
		}
	}
	srv, err := agent.New(agent.Config{
		Listen: *listen, Token: token,
		TLSCertFile: *tlsCert, TLSKeyFile: *tlsKey, ClientCAFile: *clientCA,
		MTLSOnly: *mtlsOnly, InsecureHTTP: *insecure,
		StateDir: absState, Execute: !*noExec, MaxConcurrent: *maxConc, MaxRetries: *maxRetries,
		CancelGrace: *grace, Python: *python,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "FAIL:", err)
		return 2
	}
	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		fmt.Fprintln(os.Stderr, "FAIL:", err)
		return 1
	}
	exec := fmt.Sprintf("running jobs, %d at a time, state in %s", *maxConc, filepath.Base(absState))
	if *noExec {
		exec = "queue only: jobs are NOT executed (--no-exec)"
	}
	fmt.Fprintf(os.Stderr, "forgectl agent %s listening on %s (%s; %s)\n",
		agent.Version, ln.Addr(), describeSecurity(token, *tlsCert != "", *clientCA != "", *mtlsOnly), exec)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// SIGHUP re-reads the certificate and key (rotation without a restart). It is never delivered on Windows,
	// where the 10-minute periodic reload is the only trigger.
	hup := make(chan os.Signal, 1)
	signal.Notify(hup, syscall.SIGHUP)
	defer signal.Stop(hup)
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case <-hup:
				_ = srv.ReloadCerts() // result is logged by the agent, without file paths
			}
		}
	}()

	if err := srv.Serve(ctx, ln); err != nil {
		fmt.Fprintln(os.Stderr, "FAIL:", err)
		return 1
	}
	return 0
}

func describeSecurity(token string, tls, clientCA, mtlsOnly bool) string {
	var parts []string
	switch {
	case tls && clientCA:
		parts = append(parts, "mutual TLS")
	case tls:
		parts = append(parts, "TLS")
	default:
		parts = append(parts, "plain HTTP")
	}
	switch {
	case mtlsOnly:
		parts = append(parts, "client certificate required")
	case token != "":
		parts = append(parts, "bearer token required")
	default:
		parts = append(parts, "no auth, loopback only")
	}
	return strings.Join(parts, ", ")
}
