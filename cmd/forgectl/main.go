// forgectl: kubectl-style CLI for tinyforge. Phase 1 supports offline `validate` and `plan` only.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"tinyforge.dev/forgectl/internal/doctor"
	"tinyforge.dev/forgectl/internal/jobspec"
	"tinyforge.dev/forgectl/internal/plan"
)

const usage = `forgectl: tinyforge control CLI (phase 1, offline only)

Usage:
  forgectl validate -f job.yaml    check a TrainingJob against the schema
  forgectl plan -f job.yaml        validate, show the resolved spec and any warnings
  forgectl doctor [--json]         check this machine: GPU, RAM, disk, Docker (exit 1 if any error)
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
