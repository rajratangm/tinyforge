// Package doctor runs forgectl's local single-node health check (phase 1).
//
// Scope is deliberately small and stated in finding FG099: GPU inventory/health via nvidia-smi, host RAM/disk,
// and Docker + NVIDIA runtime presence. Linux-only deep checks (XID, ECC, NVLink, PCIe AER, NUMA, NIC, SMART,
// kernel/sysctl, time sync, k8s) are NOT implemented yet.
//
// Everything that touches the machine goes through Options (command runner, system probe, file stat) so the
// logic is unit-testable with fixtures.
package doctor

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

type Level string

const (
	Info  Level = "info"
	Warn  Level = "warn"
	Error Level = "error"
)

// Finding mirrors the Python Diagnostic (code, level, message, fix).
type Finding struct {
	Code    string `json:"code"`
	Level   Level  `json:"level"`
	Message string `json:"message"`
	Fix     string `json:"fix,omitempty"`
}

type Report struct {
	System   SysInfo   `json:"system"`
	GPUs     []GPU     `json:"gpus"`
	Findings []Finding `json:"findings"`
}

// add records a finding; the fix text is optional (info findings usually have none).
func (r *Report) add(code string, lvl Level, msg string, fix ...string) {
	r.Findings = append(r.Findings, Finding{Code: code, Level: lvl, Message: msg, Fix: strings.Join(fix, " ")})
}

func (r *Report) Count(l Level) int {
	n := 0
	for _, f := range r.Findings {
		if f.Level == l {
			n++
		}
	}
	return n
}

func (r *Report) HasErrors() bool { return r.Count(Error) > 0 }

func (r *Report) JSON() ([]byte, error) { return json.MarshalIndent(r, "", "  ") }

// Runner abstracts process execution so tests can inject fixture output.
type Runner interface {
	LookPath(file string) (string, error)
	Run(ctx context.Context, name string, args ...string) (string, error)
}

type execRunner struct{ timeout time.Duration }

func (e execRunner) LookPath(file string) (string, error) { return exec.LookPath(file) }

func (e execRunner) Run(ctx context.Context, name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).CombinedOutput()
	return string(out), err
}

type Options struct {
	Runner  Runner
	Sys     func(workdir string) SysInfo
	Workdir string
	GOOS    string
	Home    string
	Stat    func(path string) bool
}

func DefaultOptions() Options {
	wd, _ := os.Getwd()
	home, _ := os.UserHomeDir()
	return Options{
		Runner:  execRunner{timeout: 20 * time.Second},
		Sys:     ReadSys,
		Workdir: wd,
		GOOS:    runtime.GOOS,
		Home:    home,
		Stat:    func(p string) bool { _, err := os.Stat(p); return err == nil },
	}
}

const gb = 1024 * 1024 * 1024

// Run executes every check and returns the report. It never panics on missing tools: absence is a finding.
func Run(ctx context.Context, o Options) *Report {
	r := &Report{GPUs: []GPU{}}
	r.System = o.Sys(o.Workdir)
	r.System.OS, r.System.Arch = o.GOOS, runtime.GOARCH
	checkHost(r)
	checkGPUs(ctx, o, r)
	checkDocker(ctx, o, r)
	r.add("FG099", Info,
		"Phase 1 checks only: GPU inventory/health via nvidia-smi, RAM, disk, Docker. NOT checked yet: XID and ECC "+
			"errors, NVLink, PCIe AER, NUMA/GPU affinity, NIC/InfiniBand, SMART/NVMe wear, kernel/sysctl, time sync, "+
			"Kubernetes, security posture.",
		"See OBJECTIVES.md section P (CLUSTER DOCTOR) for the planned checks.")
	return r
}

func checkHost(r *Report) {
	s := r.System
	if s.RAMError == "" && s.RAMTotalBytes > 0 {
		if s.RAMTotalBytes < 8*gb {
			r.add("FG009", Warn, fmt.Sprintf("Only %.1f GB system RAM; keep datasets and batch sizes small.",
				float64(s.RAMTotalBytes)/gb), "Close other applications or use a machine with more RAM.")
		}
		if s.RAMAvailBytes < 2*gb {
			r.add("FG010", Warn, fmt.Sprintf("Only %.1f GB RAM currently available; jobs may swap or be killed.",
				float64(s.RAMAvailBytes)/gb), "Close memory-heavy applications before training.")
		}
	}
	if s.DiskError == "" && s.DiskKnown {
		switch {
		case s.DiskFreeBytes < 2*gb:
			r.add("FG008", Error, fmt.Sprintf("Only %.1f GB free disk in the working directory; checkpoints "+
				"and model downloads will fail.", float64(s.DiskFreeBytes)/gb), "Free disk space or run from another volume.")
		case s.DiskFreeBytes < 10*gb:
			r.add("FG008", Warn, fmt.Sprintf("Only %.1f GB free disk in the working directory; model "+
				"downloads and checkpoints need several GB.", float64(s.DiskFreeBytes)/gb), "Free disk space or run from another volume.")
		}
	}
	if s.RAMError != "" {
		r.add("FG016", Info, "Could not read RAM size: "+s.RAMError)
	}
	if s.DiskError != "" {
		r.add("FG016", Info, "Could not read free disk space: "+s.DiskError)
	}
}

func checkGPUs(ctx context.Context, o Options, r *Report) {
	smi, err := o.Runner.LookPath("nvidia-smi")
	if err != nil {
		r.add("FG001", Warn, "No NVIDIA driver/GPU found (nvidia-smi is not on PATH); training and GPU "+
			"inference are unavailable, CPU only.", "Install a current NVIDIA driver; on Linux also the matching nvidia-utils package.")
		return
	}
	gpus, out, err := queryGPUs(ctx, o.Runner, smi)
	if err != nil {
		r.add("FG002", Error, "nvidia-smi is installed but the GPU query failed: "+err.Error()+snippet(out),
			"Check the driver is loaded (reboot after a driver update), then run `nvidia-smi` by hand.")
		return
	}
	if len(gpus) == 0 {
		r.add("FG001", Warn, "nvidia-smi ran but reported no GPUs.", "Check the driver and that the GPU is enabled.")
		return
	}
	r.GPUs = gpus
	for _, g := range gpus {
		r.add("FG003", Info, g.Summary())
		checkGPU(g, r)
	}
}

func checkGPU(g GPU, r *Report) {
	id := fmt.Sprintf("GPU %d (%s)", g.Index, g.Name)
	if g.PCIeWidthCur != nil && g.PCIeWidthMax != nil && *g.PCIeWidthCur < *g.PCIeWidthMax {
		r.add("FG004", Warn, fmt.Sprintf("%s PCIe link is x%d but the card supports x%d. Link speed and width "+
			"often drop at idle to save power, so only worry if this persists under load.",
			id, *g.PCIeWidthCur, *g.PCIeWidthMax),
			"Re-run `forgectl doctor` while a job is running; if still narrow, reseat the card or check the slot/riser.")
	}
	if g.ThrottleMask != nil {
		hard := *g.ThrottleMask & (ThrottleHWSlowdown | ThrottleSWThermal | ThrottleHWThermal | ThrottleHWPowerBrake)
		if hard != 0 {
			r.add("FG005", Warn, fmt.Sprintf("%s is being throttled now: %s.", id, strings.Join(ThrottleReasons(hard), ", ")),
				"Improve cooling/airflow, lower the power limit demand, or check PSU and cabling.")
		}
		if *g.ThrottleMask&ThrottleSWPowerCap != 0 {
			r.add("FG006", Info, fmt.Sprintf("%s is at its software power cap (normal under heavy load, "+
				"especially on laptops).", id))
		}
	}
	if g.TempC != nil && *g.TempC >= 85 {
		r.add("FG015", Warn, fmt.Sprintf("%s is at %.0f C.", id, *g.TempC), "Improve cooling; sustained temperatures this high cause throttling.")
	}
	if g.MemTotalMiB != nil && g.MemUsedMiB != nil && *g.MemTotalMiB > 0 && *g.MemUsedMiB / *g.MemTotalMiB >= 0.9 {
		r.add("FG007", Warn, fmt.Sprintf("%s already has %.0f of %.0f MiB VRAM in use by other processes; a job "+
			"may run out of memory.", id, *g.MemUsedMiB, *g.MemTotalMiB),
			"Close GPU applications (browsers, games, other jobs) or check `nvidia-smi` for what is using it.")
	}
}

func checkDocker(ctx context.Context, o Options, r *Report) {
	path, ok := findDocker(o)
	if !ok {
		r.add("FG011", Warn, "Docker CLI not found; containers (`forgectl up`, k8s nodes) need it.",
			"Install Docker Engine (Linux) or Docker Desktop (Windows/macOS) and make sure `docker` is on PATH.")
		return
	}
	out, err := o.Runner.Run(ctx, path, "info", "--format", "{{json .Runtimes}}")
	if err != nil {
		r.add("FG012", Warn, "Docker CLI found but the daemon is not reachable: "+strings.TrimSpace(firstLine(out+" "+err.Error())),
			"Start Docker Desktop / the docker service.")
		return
	}
	var runtimes map[string]json.RawMessage
	if err := json.Unmarshal([]byte(strings.TrimSpace(out)), &runtimes); err != nil {
		r.add("FG012", Warn, "Docker is running but its runtime list could not be parsed.")
		return
	}
	names := make([]string, 0, len(runtimes))
	for k := range runtimes {
		names = append(names, k)
	}
	sort.Strings(names)
	_, hasNvidia := runtimes["nvidia"]
	switch {
	case len(r.GPUs) > 0 && !hasNvidia:
		r.add("FG013", Warn, "Docker is running but has no `nvidia` runtime; containers cannot use the GPU. "+
			"Runtimes: "+strings.Join(names, ", "),
			"Install the NVIDIA Container Toolkit (Linux) or enable GPU support in Docker Desktop (WSL2 backend).")
	default:
		r.add("FG014", Info, "Docker is running. Runtimes: "+strings.Join(names, ", "))
	}
}

func findDocker(o Options) (string, bool) {
	if p, err := o.Runner.LookPath("docker"); err == nil {
		return p, true
	}
	if o.GOOS == "windows" && o.Home != "" {
		p := filepath.Join(o.Home, "AppData", "Local", "Programs", "DockerDesktop", "resources", "bin", "docker.exe")
		if o.Stat(p) {
			return p, true
		}
	}
	return "", false
}

func snippet(out string) string {
	out = strings.TrimSpace(out)
	if out == "" {
		return ""
	}
	if len(out) > 300 {
		out = out[:300] + "..."
	}
	return " (output: " + out + ")"
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
