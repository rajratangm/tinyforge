package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"time"

	"tinyforge.dev/forgectl/internal/netcheck"
)

const netUsage = `forgectl net check: reachability matrix (explicit host:port targets only; this is not a port scanner)

Usage:
  forgectl net check -f checks.yaml [--json] [--timeout 5s] [--total-timeout 2m] [--concurrency 8]
  forgectl net check --preset local [--json]     this machine's own endpoints on loopback (API, agent, Prometheus, Grafana, GPU exporter)

Exit code: 0 all expectations met (or only warn/info), 1 a check failed, 2 bad input. Schema: see internal/netcheck/spec.go.
`

func runNet(args []string) int { return runNetTo(args, os.Stdout, os.Stderr) }

func runNetTo(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 || args[0] != "check" {
		fmt.Fprint(stderr, netUsage)
		return 2
	}
	fs := flag.NewFlagSet("net check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("f", "", "checks file (YAML or JSON)")
	preset := fs.String("preset", "", "built-in check set: local")
	asJSON := fs.Bool("json", false, "emit the report as JSON")
	timeout := fs.Duration("timeout", 5*time.Second, "default per-check timeout")
	total := fs.Duration("total-timeout", 2*time.Minute, "limit for the whole run")
	conc := fs.Int("concurrency", 8, "checks in flight at once (1-64)")
	if err := fs.Parse(args[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "forgectl net check: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if (*file == "") == (*preset == "") {
		fmt.Fprintln(stderr, "forgectl net check: give exactly one of -f <file> or --preset local")
		return 2
	}
	if *timeout <= 0 || *total <= 0 || *conc < 1 || *conc > 64 {
		fmt.Fprintln(stderr, "forgectl net check: --timeout and --total-timeout must be positive and --concurrency 1-64")
		return 2
	}

	var checks []netcheck.Check
	switch {
	case *preset != "":
		if *preset != "local" {
			fmt.Fprintf(stderr, "forgectl net check: unknown preset %q (only \"local\")\n", *preset)
			return 2
		}
		checks = netcheck.LocalPreset()
	default:
		var err error
		if checks, err = netcheck.Load(*file); err != nil {
			fmt.Fprintln(stderr, "FAIL:", err)
			return 2
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	opts := netcheck.DefaultOptions()
	opts.Timeout, opts.TotalTimeout, opts.Concurrency = *timeout, *total, *conc
	rep := netcheck.Run(ctx, checks, opts)

	if *asJSON {
		b, err := rep.JSON()
		if err != nil {
			fmt.Fprintln(stderr, "FAIL:", err)
			return 2
		}
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprint(stdout, netcheck.Render(rep))
	}
	if rep.Failed() {
		return 1
	}
	return 0
}
