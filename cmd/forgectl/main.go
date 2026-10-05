// forgectl: kubectl-style CLI for tinyforge. Phase 1 supports offline `validate` and `plan` only.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"tinyforge.dev/forgectl/internal/doctor"
	"tinyforge.dev/forgectl/internal/jobspec"
	"tinyforge.dev/forgectl/internal/plan"
	"tinyforge.dev/forgectl/internal/runner"
)

const usage = `forgectl: tinyforge control CLI (phase 1, offline only)

Usage:
  forgectl validate -f job.yaml    check a TrainingJob against the schema
  forgectl plan -f job.yaml        validate, show the resolved spec and any warnings
  forgectl doctor [--json]         check this machine: GPU, RAM, disk, Docker (exit 1 if any error)
  forgectl run -f job.yaml         validate, then run the Python worker locally and stream its events
      [--out DIR] [--python PATH] [--dry-run] [--json]   (exit code = the worker's, see spec/worker-contract.md)
  forgectl agent [--listen ADDR] [--token-file F]   node agent: /healthz /readyz /v1/node /metrics /v1/jobs (stub queue)
  forgectl net check (-f checks.yaml | --preset local) [--json]   reachability matrix: dns, tcp, tls, http, egress-deny
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	switch os.Args[1] {
	case "validate":
		os.Exit(runValidate(os.Args[2:]))
	case "plan":
		os.Exit(runPlan(os.Args[2:]))
	case "doctor":
		os.Exit(runDoctor(os.Args[2:]))
	case "run":
		os.Exit(runRun(os.Args[2:]))
	case "agent":
		os.Exit(runAgent(os.Args[2:]))
	case "net":
		os.Exit(runNet(os.Args[2:]))
	case "-h", "--help", "help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "forgectl: unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}
}

func fileFlag(name string, args []string) (string, bool) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	f := fs.String("f", "", "path to a TrainingJob YAML/JSON file")
	if err := fs.Parse(args); err != nil {
		return "", false
	}
	if *f == "" {
		fmt.Fprintf(os.Stderr, "forgectl %s: -f <file> is required\n", name)
		return "", false
	}
	return *f, true
}

func runValidate(args []string) int {
	path, ok := fileFlag("validate", args)
	if !ok {
		return 2
	}
	j, err := jobspec.Load(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "FAIL:", err)
		return 2 // matches the worker contract: 2 = spec invalid
	}
	fmt.Printf("OK: TrainingJob %s/%s is valid\n", j.Metadata.Namespace, j.Metadata.Name)
	return 0
}

func runDoctor(args []string) int {
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit the report as JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	rep := doctor.Run(context.Background(), doctor.DefaultOptions())
	if *asJSON {
		b, err := rep.JSON()
		if err != nil {
			fmt.Fprintln(os.Stderr, "FAIL:", err)
			return 2
		}
		fmt.Println(string(b))
	} else {
		fmt.Print(doctor.Render(rep))
	}
	if rep.HasErrors() {
		return 1
	}
	return 0
}

func runRun(args []string) int {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	f := fs.String("f", "", "path to a TrainingJob YAML/JSON file")
	out := fs.String("out", "", "worker output dir (default runs/<metadata.name>)")
	python := fs.String("python", "", "Python interpreter (default: $FORGECTL_PYTHON, a project .venv, then PATH)")
	dry := fs.Bool("dry-run", false, "validate and print the worker's resolved config; train nothing")
	asJSON := fs.Bool("json", false, "pass the worker's raw JSON lines through on stdout")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *f == "" {
		fmt.Fprintln(os.Stderr, "forgectl run: -f <file> is required")
		return 2
	}
	sigs := make(chan os.Signal, 2)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	return runner.Run(runner.Options{
		SpecPath: *f, Out: *out, Python: *python, DryRun: *dry, JSON: *asJSON,
		Stdout: os.Stdout, Stderr: os.Stderr, Signals: sigs,
	})
}

func runPlan(args []string) int {
	path, ok := fileFlag("plan", args)
	if !ok {
		return 2
	}
	j, err := jobspec.Load(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "FAIL:", err)
		return 2
	}
	fmt.Print(plan.Render(plan.Build(j)))
	return 0
}
