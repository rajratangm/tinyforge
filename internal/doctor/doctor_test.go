package doctor

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixture(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// fakeRunner serves canned tool output. A tool absent from `paths` is "not installed".
type fakeRunner struct {
	paths map[string]string
	run   func(name string, args []string) (string, error)
	calls []string
}

func (f *fakeRunner) LookPath(file string) (string, error) {
	if p, ok := f.paths[file]; ok {
		return p, nil
	}
	return "", errors.New("not found")
}

func (f *fakeRunner) Run(_ context.Context, name string, args ...string) (string, error) {
	f.calls = append(f.calls, name+" "+strings.Join(args, " "))
	return f.run(name, args)
}

func goodSys(string) SysInfo {
	return SysInfo{CPUs: 8, RAMTotalBytes: 16 * gb, RAMAvailBytes: 8 * gb, DiskFreeBytes: 200 * gb, DiskKnown: true}
}

func opts(rn Runner) Options {
	return Options{Runner: rn, Sys: goodSys, Workdir: ".", GOOS: "linux", Home: "/home/x",
		Stat: func(string) bool { return false }}
}

func codes(r *Report) map[string]Level {
	m := map[string]Level{}
	for _, f := range r.Findings {
		m[f.Code] = f.Level
	}
	return m
}

func TestParseGPUs(t *testing.T) {
	t.Run("single", func(t *testing.T) {
		g, err := ParseGPUs(fixture(t, "smi_single.csv"))
		if err != nil || len(g) != 1 {
			t.Fatalf("err=%v gpus=%v", err, g)
		}
		if g[0].Name != "NVIDIA GeForce RTX 3050 Ti Laptop GPU" || *g[0].MemTotalMiB != 4096 ||
			g[0].Driver != "560.94" || *g[0].ThrottleMask != 1 || *g[0].PCIeWidthMax != 8 {
			t.Fatalf("bad parse: %+v", g[0])
		}
	})
	t.Run("multi", func(t *testing.T) {
		g, err := ParseGPUs(fixture(t, "smi_multi_bad.csv"))
		if err != nil || len(g) != 2 || g[1].Index != 1 || *g[1].ThrottleMask != 0x68 {
			t.Fatalf("err=%v gpus=%v", err, g)
		}
	})
	t.Run("N/A values become nil, not zero", func(t *testing.T) {
		g, err := ParseGPUs(fixture(t, "smi_na.csv"))
		if err != nil || len(g) != 1 {
			t.Fatalf("err=%v", err)
		}
		x := g[0]
		if x.TempC != nil || x.PowerW != nil || x.PowerLimitW != nil || x.ThrottleMask != nil ||
			x.PCIeGenCur != nil || x.PCIeWidthMax != nil {
			t.Fatalf("N/A fields should be nil: %+v", x)
		}
		if *x.MemTotalMiB != 15360 {
			t.Fatal("known fields lost")
		}
		if !strings.Contains(x.Summary(), "Tesla T4") {
			t.Fatal("summary should still render")
		}
	})
	t.Run("errors", func(t *testing.T) {
		for name, in := range map[string]string{
			"wrong columns": "0, GPU, 1, 2\n",
			"bad index":     strings.Replace(fixture(t, "smi_single.csv"), "0, NVIDIA", "x, NVIDIA", 1),
		} {
			if _, err := ParseGPUs(in); err == nil {
				t.Errorf("%s: expected error", name)
			}
		}
	})
	t.Run("empty output is zero GPUs", func(t *testing.T) {
		if g, err := ParseGPUs("\n  \n"); err != nil || len(g) != 0 {
			t.Fatalf("g=%v err=%v", g, err)
		}
	})
}

func TestThrottleReasons(t *testing.T) {
	got := strings.Join(ThrottleReasons(0x68), ",")
	for _, want := range []string{"hardware slowdown", "software thermal slowdown", "hardware thermal slowdown"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in %q", want, got)
		}
	}
	if len(ThrottleReasons(0)) != 0 {
		t.Error("mask 0 should give no reasons")
	}
}

func TestParseMeminfo(t *testing.T) {
	total, avail, err := parseMeminfo(fixture(t, "meminfo.txt"))
	if err != nil || total != 16384000*1024 || avail != 8192000*1024 {
		t.Fatalf("total=%d avail=%d err=%v", total, avail, err)
	}
	if _, _, err := parseMeminfo("MemTotal: 1 kB\n"); err == nil {
		t.Fatal("missing MemAvailable should error")
	}
}

func smiRunner(t *testing.T, csv string, docker string) *fakeRunner {
	return &fakeRunner{
		paths: map[string]string{"nvidia-smi": "/usr/bin/nvidia-smi", "docker": "/usr/bin/docker"},
		run: func(name string, args []string) (string, error) {
			if strings.HasSuffix(name, "nvidia-smi") {
				return csv, nil
			}
			return docker, nil
		},
	}
}

func TestRunHealthyMachine(t *testing.T) {
	r := Run(context.Background(), opts(smiRunner(t, fixture(t, "smi_single.csv"), fixture(t, "docker_runtimes_nvidia.json"))))
	if r.HasErrors() || r.Count(Warn) != 0 {
		t.Fatalf("healthy machine should be clean: %+v", r.Findings)
	}
	c := codes(r)
	for _, want := range []string{"FG003", "FG014", "FG099"} {
		if _, ok := c[want]; !ok {
			t.Errorf("missing %s: %v", want, c)
		}
	}
	if len(r.GPUs) != 1 {
		t.Fatal("report should list the GPU")
	}
}

func TestRunUnhealthyMultiGPU(t *testing.T) {
	r := Run(context.Background(), opts(smiRunner(t, fixture(t, "smi_multi_bad.csv"), fixture(t, "docker_runtimes_nvidia.json"))))
	c := codes(r)
	for _, want := range []string{"FG004", "FG005", "FG007", "FG015"} {
		if c[want] != Warn {
			t.Errorf("want warn %s, got %v", want, c)
		}
	}
	for _, f := range r.Findings {
		if (f.Code == "FG004" || f.Code == "FG005" || f.Code == "FG007" || f.Code == "FG015") && !strings.Contains(f.Message, "GPU 1") {
			t.Errorf("%s should point at GPU 1 only, got %q", f.Code, f.Message)
		}
	}
	if r.HasErrors() {
		t.Fatal("these are warnings, not errors")
	}
	for _, f := range r.Findings {
		if f.Code == "FG004" && !strings.Contains(f.Message, "idle") {
			t.Error("PCIe warning must carry the idle caveat")
		}
	}
}

func TestRunNoNvidiaSmi(t *testing.T) {
	rn := &fakeRunner{paths: map[string]string{}, run: func(string, []string) (string, error) { return "", errors.New("x") }}
	r := Run(context.Background(), opts(rn))
	c := codes(r)
	if c["FG001"] != Warn || c["FG011"] != Warn {
		t.Fatalf("want FG001+FG011 warnings: %v", c)
	}
	if r.HasErrors() {
		t.Fatal("a machine without a GPU or docker is a warning, not a crash or error")
	}
}

func TestRunNvidiaSmiFails(t *testing.T) {
	rn := &fakeRunner{
		paths: map[string]string{"nvidia-smi": "nvidia-smi"},
		run: func(string, []string) (string, error) {
			return "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.", errors.New("exit status 9")
		},
	}
	r := Run(context.Background(), opts(rn))
	if codes(r)["FG002"] != Error || !r.HasErrors() {
		t.Fatalf("want FG002 error: %+v", r.Findings)
	}
}

func TestRunNvidiaSmiGarbageOutput(t *testing.T) {
	r := Run(context.Background(), opts(smiRunner(t, "this is not csv", "{}")))
	if codes(r)["FG002"] != Error {
		t.Fatalf("unparseable output should be FG002: %+v", r.Findings)
	}
}

func TestThrottleFieldFallback(t *testing.T) {
	rn := &fakeRunner{paths: map[string]string{"nvidia-smi": "nvidia-smi"}}
	rn.run = func(name string, args []string) (string, error) {
		if strings.Contains(args[0], "clocks_throttle_reasons") {
			return fixture(t, "smi_badfield.txt"), errors.New("exit status 2")
		}
		return fixture(t, "smi_single.csv"), nil
	}
	r := Run(context.Background(), opts(rn))
	if len(r.GPUs) != 1 || codes(r)["FG002"] != "" {
		t.Fatalf("fallback to clocks_event_reasons failed: %+v", r.Findings)
	}
	if len(rn.calls) < 2 || !strings.Contains(rn.calls[1], "clocks_event_reasons.active") {
		t.Fatalf("second query should use the new field name: %v", rn.calls)
	}
}

func TestDockerChecks(t *testing.T) {
	smi := fixture(t, "smi_single.csv")
	t.Run("daemon down", func(t *testing.T) {
		rn := smiRunner(t, smi, "")
		inner := rn.run
		rn.run = func(name string, a []string) (string, error) {
			if strings.HasSuffix(name, "docker") {
				return "error during connect: daemon not running", errors.New("exit status 1")
			}
			return inner(name, a)
		}
		if codes(Run(context.Background(), opts(rn)))["FG012"] != Warn {
			t.Fatal("want FG012")
		}
	})
	t.Run("no nvidia runtime with a GPU", func(t *testing.T) {
		r := Run(context.Background(), opts(smiRunner(t, smi, fixture(t, "docker_runtimes_plain.json"))))
		if codes(r)["FG013"] != Warn {
			t.Fatalf("want FG013: %v", codes(r))
		}
	})
	t.Run("no nvidia runtime without a GPU is fine", func(t *testing.T) {
		rn := &fakeRunner{paths: map[string]string{"docker": "docker"},
			run: func(string, []string) (string, error) { return fixture(t, "docker_runtimes_plain.json"), nil }}
		if _, bad := codes(Run(context.Background(), opts(rn)))["FG013"]; bad {
			t.Fatal("FG013 should need a GPU")
		}
	})
	t.Run("windows Docker Desktop outside PATH", func(t *testing.T) {
		want := filepath.Join("C:/Users/me", "AppData", "Local", "Programs", "DockerDesktop", "resources", "bin", "docker.exe")
		rn := &fakeRunner{paths: map[string]string{},
			run: func(name string, _ []string) (string, error) {
				if name != want {
					t.Errorf("ran %q, want %q", name, want)
				}
				return fixture(t, "docker_runtimes_plain.json"), nil
			}}
		o := opts(rn)
		o.GOOS, o.Home = "windows", "C:/Users/me"
		o.Stat = func(p string) bool { return p == want }
		if _, ok := codes(Run(context.Background(), o))["FG014"]; !ok {
			t.Fatal("should have found docker at the Docker Desktop path")
		}
	})
	t.Run("windows path only used on windows", func(t *testing.T) {
		rn := &fakeRunner{paths: map[string]string{}, run: func(string, []string) (string, error) { return "", nil }}
		o := opts(rn)
		o.Stat = func(string) bool { return true }
		if codes(Run(context.Background(), o))["FG011"] != Warn {
			t.Fatal("linux must not look in the Windows path")
		}
	})
}

func TestHostChecks(t *testing.T) {
	cases := map[string]struct {
		sys  SysInfo
		code string
		lvl  Level
	}{
		"low disk warn":  {SysInfo{RAMTotalBytes: 16 * gb, RAMAvailBytes: 8 * gb, DiskFreeBytes: 5 * gb, DiskKnown: true}, "FG008", Warn},
		"disk full err":  {SysInfo{RAMTotalBytes: 16 * gb, RAMAvailBytes: 8 * gb, DiskFreeBytes: 1 * gb, DiskKnown: true}, "FG008", Error},
		"low total RAM":  {SysInfo{RAMTotalBytes: 4 * gb, RAMAvailBytes: 3 * gb, DiskFreeBytes: 100 * gb, DiskKnown: true}, "FG009", Warn},
		"low avail RAM":  {SysInfo{RAMTotalBytes: 16 * gb, RAMAvailBytes: 1 * gb, DiskFreeBytes: 100 * gb, DiskKnown: true}, "FG010", Warn},
		"disk read fail": {SysInfo{RAMTotalBytes: 16 * gb, RAMAvailBytes: 8 * gb, DiskError: "boom"}, "FG016", Info},
		"ram read fail":  {SysInfo{RAMError: "boom", DiskFreeBytes: 100 * gb, DiskKnown: true}, "FG016", Info},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			o := opts(&fakeRunner{paths: map[string]string{}, run: func(string, []string) (string, error) { return "", nil }})
			o.Sys = func(string) SysInfo { return c.sys }
			if got := codes(Run(context.Background(), o))[c.code]; got != c.lvl {
				t.Fatalf("%s: want %s %s, got %q", name, c.code, c.lvl, got)
			}
		})
	}
	t.Run("unreadable host never produces false disk/RAM warnings", func(t *testing.T) {
		o := opts(&fakeRunner{paths: map[string]string{}, run: func(string, []string) (string, error) { return "", nil }})
		o.Sys = func(string) SysInfo { return SysInfo{RAMError: "x", DiskError: "x"} }
		c := codes(Run(context.Background(), o))
		for _, bad := range []string{"FG008", "FG009", "FG010"} {
			if _, ok := c[bad]; ok {
				t.Errorf("%s fired although the value is unknown", bad)
			}
		}
	})
}

func TestRenderAndJSON(t *testing.T) {
	r := Run(context.Background(), opts(smiRunner(t, fixture(t, "smi_multi_bad.csv"), fixture(t, "docker_runtimes_plain.json"))))
	out := Render(r)
	for _, s := range []string{"forgectl doctor", "FG004", "fix:", "error(s)", "warning(s)", "NVIDIA A100"} {
		if !strings.Contains(out, s) {
			t.Errorf("render missing %q:\n%s", s, out)
		}
	}
	b, err := r.JSON()
	if err != nil {
		t.Fatal(err)
	}
	var back Report
	if err := json.Unmarshal(b, &back); err != nil || len(back.GPUs) != 2 || len(back.Findings) != len(r.Findings) {
		t.Fatalf("JSON round trip failed: %v", err)
	}
	if !strings.Contains(string(b), `"level": "warn"`) {
		t.Error("levels should serialise as strings")
	}
}
