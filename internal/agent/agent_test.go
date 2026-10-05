package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"tinyforge.dev/forgectl/internal/doctor"
)

const validSpec = `
apiVersion: tinyforge.dev/v1alpha1
kind: TrainingJob
metadata: {name: t1}
spec:
  method: lora
  model: {base: org/model}
  data: {source: org/data}
`

func f64(v float64) *float64 { return &v }

func fakeReport() *doctor.Report {
	return &doctor.Report{
		System: doctor.SysInfo{OS: "linux", Arch: "amd64"},
		GPUs: []doctor.GPU{
			{Index: 0, Name: "GPU A", MemTotalMiB: f64(4096), MemUsedMiB: f64(1024), TempC: f64(60)},
			{Index: 1, Name: "GPU B", MemTotalMiB: f64(8192)}, // used and temp reported as N/A
		},
		Findings: []doctor.Finding{
			{Code: "FG001", Level: doctor.Info, Message: "m"},
			{Code: "FG002", Level: doctor.Warn, Message: "m"},
			{Code: "FG003", Level: doctor.Warn, Message: "m"},
		},
	}
}

type clock struct{ t time.Time }

func (c *clock) now() time.Time { return c.t }

func newTestServer(t *testing.T, mut func(*Config)) (*Server, *clock) {
	t.Helper()
	clk := &clock{t: time.Date(2026, 10, 5, 12, 0, 0, 0, time.UTC)}
	cfg := Config{
		Hostname: "testhost", Now: clk.now, Log: log.New(io.Discard, "", 0),
		Doctor: func(context.Context) *doctor.Report { return fakeReport() },
	}
	if mut != nil {
		mut(&cfg)
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return s, clk
}

func do(s *Server, method, path, body string, hdr map[string]string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	return rec
}

func bearer(tok string) map[string]string { return map[string]string{"Authorization": "Bearer " + tok} }

func TestHealthzOpenEvenWithToken(t *testing.T) {
	s, _ := newTestServer(t, func(c *Config) { c.Token = "secret" })
	if rec := do(s, "GET", "/healthz", "", nil); rec.Code != 200 {
		t.Fatalf("healthz = %d", rec.Code)
	}
}

func TestReadyzWaitsForFirstReport(t *testing.T) {
	s, _ := newTestServer(t, nil)
	if rec := do(s, "GET", "/readyz", "", nil); rec.Code != 503 {
		t.Fatalf("readyz before refresh = %d, want 503", rec.Code)
	}
	if rec := do(s, "GET", "/v1/node", "", nil); rec.Code != 503 {
		t.Fatalf("node before refresh = %d, want 503", rec.Code)
	}
	s.Refresh(context.Background())
	if rec := do(s, "GET", "/readyz", "", nil); rec.Code != 200 {
		t.Fatalf("readyz after refresh = %d, want 200", rec.Code)
	}
}

func TestAuthRequiredWhenTokenSet(t *testing.T) {
	s, _ := newTestServer(t, func(c *Config) { c.Token = "secret" })
	s.Refresh(context.Background())
	endpoints := []struct{ method, path string }{
		{"GET", "/v1/node"}, {"GET", "/metrics"}, {"GET", "/v1/jobs"}, {"POST", "/v1/jobs"}, {"GET", "/v1/jobs/job-x"},
		{"GET", "/v1/jobs/job-x/events"}, {"POST", "/v1/jobs/job-x/cancel"}, {"DELETE", "/v1/jobs/job-x"},
	}
	for _, e := range endpoints {
		if rec := do(s, e.method, e.path, validSpec, nil); rec.Code != 401 {
			t.Errorf("%s %s without token = %d, want 401", e.method, e.path, rec.Code)
		} else if rec.Header().Get("WWW-Authenticate") == "" {
			t.Errorf("%s %s: 401 without WWW-Authenticate", e.method, e.path)
		}
		if rec := do(s, e.method, e.path, validSpec, bearer("wrong")); rec.Code != 403 {
			t.Errorf("%s %s with wrong token = %d, want 403", e.method, e.path, rec.Code)
		}
		if rec := do(s, e.method, e.path, validSpec, map[string]string{"Authorization": "Basic secret"}); rec.Code != 401 {
			t.Errorf("%s %s with Basic scheme = %d, want 401", e.method, e.path, rec.Code)
		}
	}
	if rec := do(s, "GET", "/v1/node", "", bearer("secret")); rec.Code != 200 {
		t.Errorf("node with right token = %d, want 200", rec.Code)
	}
	if rec := do(s, "GET", "/metrics", "", map[string]string{"Authorization": "bearer secret"}); rec.Code != 200 {
		t.Errorf("metrics with lowercase scheme = %d, want 200", rec.Code)
	}
}

func TestNoTokenLoopbackIsOpen(t *testing.T) {
	s, _ := newTestServer(t, nil)
	s.Refresh(context.Background())
	if rec := do(s, "GET", "/v1/node", "", nil); rec.Code != 200 {
		t.Fatalf("node = %d, want 200", rec.Code)
	}
}

func TestNodeJSON(t *testing.T) {
	s, clk := newTestServer(t, nil)
	s.Refresh(context.Background())
	clk.t = clk.t.Add(90 * time.Second)
	rec := do(s, "GET", "/v1/node", "", nil)
	var got struct {
		Hostname         string  `json:"hostname"`
		OS               string  `json:"os"`
		Arch             string  `json:"arch"`
		AgentVersion     string  `json:"agent_version"`
		UptimeSeconds    float64 `json:"uptime_seconds"`
		ReportAgeSeconds float64 `json:"report_age_seconds"`
		Report           struct {
			GPUs     []map[string]any `json:"gpus"`
			Findings []map[string]any `json:"findings"`
		} `json:"report"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Hostname != "testhost" || got.OS != "linux" || got.Arch != "amd64" || got.AgentVersion != Version ||
		got.UptimeSeconds != 90 || got.ReportAgeSeconds != 90 || len(got.Report.GPUs) != 2 || len(got.Report.Findings) != 3 {
		t.Fatalf("unexpected node info: %+v", got)
	}
}

func TestSubmitInvalidSpecIs400(t *testing.T) {
	s, _ := newTestServer(t, nil)
	for name, body := range map[string]string{
		"bad method": strings.Replace(validSpec, "method: lora", "method: nope", 1),
		"not yaml":   "::: not yaml :::",
		"empty":      "",
	} {
		rec := do(s, "POST", "/v1/jobs", body, nil)
		if rec.Code != 400 {
			t.Errorf("%s: status %d, want 400", name, rec.Code)
		}
		if !strings.Contains(rec.Body.String(), `"error"`) {
			t.Errorf("%s: body lacks error field: %s", name, rec.Body.String())
		}
	}
	if n := s.jobs.count(); n != 0 {
		t.Fatalf("invalid specs were queued: %d", n)
	}
}

func TestSubmitListGetJob(t *testing.T) {
	s, _ := newTestServer(t, nil)
	rec := do(s, "POST", "/v1/jobs", validSpec, nil)
	if rec.Code != 202 {
		t.Fatalf("submit = %d: %s", rec.Code, rec.Body.String())
	}
	var job Job
	if err := json.Unmarshal(rec.Body.Bytes(), &job); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(job.ID, "job-") || job.Name != "t1" || job.Namespace != "default" ||
		job.Status != "queued" || job.Attempts != 0 || job.Started != nil || job.Finished != nil {
		t.Fatalf("unexpected job: %+v", job)
	}
	var list struct{ Jobs []Job }
	rec = do(s, "GET", "/v1/jobs", "", nil)
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil || len(list.Jobs) != 1 || list.Jobs[0].ID != job.ID {
		t.Fatalf("list = %s (err %v)", rec.Body.String(), err)
	}
	if rec = do(s, "GET", "/v1/jobs/"+job.ID, "", nil); rec.Code != 200 {
		t.Fatalf("get = %d", rec.Code)
	}
	if rec = do(s, "GET", "/v1/jobs/job-nope", "", nil); rec.Code != 404 {
		t.Fatalf("get unknown = %d, want 404", rec.Code)
	}
	if strings.Contains(rec.Body.String(), "stub") {
		t.Fatalf("response still carries the stub marker: %s", rec.Body.String())
	}
	if rec = do(s, "DELETE", "/v1/jobs/"+job.ID, "", nil); rec.Code != 409 {
		t.Fatalf("delete of a queued job = %d, want 409 (cancel it first)", rec.Code)
	}
	if rec = do(s, "DELETE", "/v1/jobs/job-nope", "", nil); rec.Code != 404 {
		t.Fatalf("delete unknown = %d, want 404", rec.Code)
	}
}

func TestBodyLimit(t *testing.T) {
	s, _ := newTestServer(t, func(c *Config) { c.MaxBody = 256 })
	big := validSpec + "# " + strings.Repeat("x", 1024) + "\n"
	if rec := do(s, "POST", "/v1/jobs", big, nil); rec.Code != 413 {
		t.Fatalf("oversized body = %d, want 413", rec.Code)
	}
	if rec := do(s, "POST", "/v1/jobs", validSpec, nil); rec.Code != 202 {
		t.Fatalf("small body = %d, want 202", rec.Code)
	}
	if got := defaultMaxBody; got != 1<<20 {
		t.Fatalf("default body limit = %d, want 1 MiB", got)
	}
}

func TestJobQueueCap(t *testing.T) {
	s, _ := newTestServer(t, func(c *Config) { c.MaxJobs = 2 })
	for i := 0; i < 2; i++ {
		if rec := do(s, "POST", "/v1/jobs", validSpec, nil); rec.Code != 202 {
			t.Fatalf("submit %d = %d", i, rec.Code)
		}
	}
	if rec := do(s, "POST", "/v1/jobs", validSpec, nil); rec.Code != 429 {
		t.Fatalf("over cap = %d, want 429", rec.Code)
	}
}

// parseMetrics is a tiny exposition-format parser: it rejects samples without a preceding HELP and TYPE,
// malformed lines, and bad label quoting.
func parseMetrics(t *testing.T, text string) map[string]float64 {
	t.Helper()
	out := map[string]float64{}
	help, typ, hist := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, line := range strings.Split(strings.TrimSuffix(text, "\n"), "\n") {
		switch {
		case strings.HasPrefix(line, "# HELP "):
			help[strings.Fields(line)[2]] = true
		case strings.HasPrefix(line, "# TYPE "):
			f := strings.Fields(line)
			if !help[f[2]] {
				t.Errorf("TYPE before HELP for %s", f[2])
			}
			if f[3] != "gauge" && f[3] != "counter" && f[3] != "histogram" {
				t.Errorf("bad type %q", f[3])
			}
			typ[f[2]] = true
			if f[3] == "histogram" {
				hist[f[2]] = true
			}
		case line == "":
			t.Errorf("blank line in exposition output")
		default:
			i := strings.LastIndex(line, " ")
			if i < 0 {
				t.Fatalf("malformed sample %q", line)
			}
			series, val := line[:i], line[i+1:]
			name := series
			if j := strings.Index(series, "{"); j >= 0 {
				name = series[:j]
				if !strings.HasSuffix(series, "}") || strings.Count(series[j:], `"`)%2 != 0 {
					t.Errorf("bad labels in %q", line)
				}
			}
			base := name
			for _, suf := range []string{"_bucket", "_sum", "_count"} {
				if b := strings.TrimSuffix(name, suf); b != name && hist[b] {
					base = b // histogram samples are documented under the family name
				}
			}
			if !help[base] || !typ[base] {
				t.Errorf("sample %q without HELP/TYPE", name)
			}
			var v float64
			if err := json.Unmarshal([]byte(val), &v); err != nil {
				t.Errorf("bad value in %q: %v", line, err)
			}
			if _, dup := out[series]; dup {
				t.Errorf("duplicate series %s", series)
			}
			out[series] = v
		}
	}
	return out
}

func TestMetricsFormatAndValues(t *testing.T) {
	s, clk := newTestServer(t, nil)
	do(s, "POST", "/v1/jobs", validSpec, nil)
	s.Refresh(context.Background())
	clk.t = clk.t.Add(30 * time.Second)
	rec := do(s, "GET", "/metrics", "", nil)
	if rec.Code != 200 || !strings.HasPrefix(rec.Header().Get("Content-Type"), "text/plain; version=0.0.4") {
		t.Fatalf("status %d content-type %q", rec.Code, rec.Header().Get("Content-Type"))
	}
	m := parseMetrics(t, rec.Body.String())
	want := map[string]float64{
		"forgectl_agent_up": 1,
		`forgectl_agent_info{version="` + Version + `"}`: 1,
		"forgectl_agent_uptime_seconds":                  30,
		"forgectl_agent_jobs_queued":                     1,
		"forgectl_doctor_report_age_seconds":             30,
		`forgectl_doctor_findings{severity="info"}`:      1,
		`forgectl_doctor_findings{severity="warn"}`:      2,
		`forgectl_doctor_findings{severity="error"}`:     0,
		"forgectl_node_gpu_count":                        2,
		`forgectl_node_gpu_memory_total_bytes{gpu="0"}`:  4096 * miB,
		`forgectl_node_gpu_memory_total_bytes{gpu="1"}`:  8192 * miB,
		`forgectl_node_gpu_memory_used_bytes{gpu="0"}`:   1024 * miB,
		`forgectl_node_gpu_temperature_celsius{gpu="0"}`: 60,
	}
	for k, v := range want {
		if got, ok := m[k]; !ok || got != v {
			t.Errorf("%s = %v (present %v), want %v", k, got, ok, v)
		}
	}
	// GPU 1 reported N/A for used memory and temperature: omitted, not zero.
	for _, k := range []string{`forgectl_node_gpu_memory_used_bytes{gpu="1"}`, `forgectl_node_gpu_temperature_celsius{gpu="1"}`} {
		if _, ok := m[k]; ok {
			t.Errorf("%s should be omitted when the driver reports N/A", k)
		}
	}
}

func TestMetricsBeforeFirstReport(t *testing.T) {
	s, _ := newTestServer(t, nil)
	m := parseMetrics(t, do(s, "GET", "/metrics", "", nil).Body.String())
	if m["forgectl_agent_up"] != 1 {
		t.Fatal("agent_up missing")
	}
	if _, ok := m["forgectl_node_gpu_count"]; ok {
		t.Fatal("gpu metrics must be absent until a report exists")
	}
}

func TestEscapeLabel(t *testing.T) {
	if got := escapeLabel("a\\b\"c\nd"); got != `a\\b\"c\nd` {
		t.Fatalf("escapeLabel = %q", got)
	}
}

func TestDoctorReportIsCachedNotPerRequest(t *testing.T) {
	var calls atomic.Int32
	s, _ := newTestServer(t, func(c *Config) {
		c.Doctor = func(context.Context) *doctor.Report { calls.Add(1); return fakeReport() }
	})
	s.Refresh(context.Background())
	for i := 0; i < 20; i++ {
		do(s, "GET", "/v1/node", "", nil)
		do(s, "GET", "/metrics", "", nil)
	}
	if n := calls.Load(); n != 1 {
		t.Fatalf("doctor ran %d times for 40 requests, want 1", n)
	}
}

func TestCheckListen(t *testing.T) {
	none := Security{}
	tokenOnly := Security{Token: true}
	tlsOnly := Security{TLS: true}
	tlsToken := Security{TLS: true, Token: true}
	tlsMTLS := Security{TLS: true, MTLS: true}
	insecureToken := Security{InsecureHTTP: true, Token: true}
	cases := []struct {
		addr string
		sec  Security
		ok   bool
	}{
		// Loopback is always fine, with or without anything.
		{"127.0.0.1:7070", none, true},
		{"[::1]:7070", none, true},
		{"localhost:7070", none, true},
		// Non-loopback without TLS is refused, even with a token (the old behaviour).
		{"0.0.0.0:7070", none, false},
		{":7070", none, false},
		{"[::]:7070", none, false},
		{"10.1.2.3:7070", none, false},
		{"node1.example.com:7070", none, false},
		{"0.0.0.0:7070", tokenOnly, false},
		{":7070", tokenOnly, false},
		// TLS alone is not enough: credentials are needed too.
		{"0.0.0.0:7070", tlsOnly, false},
		{"0.0.0.0:7070", tlsToken, true},
		{":7070", tlsToken, true},
		{"10.1.2.3:7070", tlsMTLS, true},
		// The escape hatch waives TLS only; a token is still required.
		{"0.0.0.0:7070", insecureToken, true},
		{"0.0.0.0:7070", Security{InsecureHTTP: true}, false},
		{"not-an-address", tlsToken, false},
	}
	for _, c := range cases {
		err := CheckListen(c.addr, c.sec)
		if (err == nil) != c.ok {
			t.Errorf("CheckListen(%q, %+v) err=%v, want ok=%v", c.addr, c.sec, err, c.ok)
		}
	}
}

func TestNewRefusesNonLoopbackWithoutTLS(t *testing.T) {
	if _, err := New(Config{Listen: "0.0.0.0:7070"}); err == nil {
		t.Fatal("New accepted a non-loopback address without TLS or a token")
	}
	if _, err := New(Config{Listen: "0.0.0.0:7070", Token: "tok"}); err == nil {
		t.Fatal("New accepted a non-loopback address with a token but no TLS")
	}
	if _, err := New(Config{Listen: "0.0.0.0:7070", Token: "tok", InsecureHTTP: true, Log: log.New(io.Discard, "", 0)}); err != nil {
		t.Fatalf("New rejected the explicit --insecure-http escape hatch with a token: %v", err)
	}
}

func TestLogsNeverContainTokenOrSpec(t *testing.T) {
	var buf bytes.Buffer
	s, _ := newTestServer(t, func(c *Config) { c.Token = "s3cr3t-token"; c.Log = log.New(&buf, "", 0) })
	do(s, "POST", "/v1/jobs", validSpec, bearer("s3cr3t-token"))
	do(s, "GET", "/v1/jobs", "", bearer("wrong-token-value"))
	out := buf.String()
	if out == "" {
		t.Fatal("expected request log lines")
	}
	for _, leak := range []string{"s3cr3t-token", "wrong-token-value", "org/model"} {
		if strings.Contains(out, leak) {
			t.Fatalf("log leaked %q:\n%s", leak, out)
		}
	}
}

func TestServeAndGracefulShutdown(t *testing.T) {
	s, _ := newTestServer(t, func(c *Config) { c.RefreshEvery = 20 * time.Millisecond })
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- s.Serve(ctx, ln) }()

	base := "http://" + ln.Addr().String()
	deadline := time.Now().Add(5 * time.Second)
	for {
		resp, err := http.Get(base + "/readyz")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				break
			}
		}
		if time.Now().After(deadline) {
			t.Fatal("agent never became ready")
		}
		time.Sleep(10 * time.Millisecond)
	}
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Serve returned %v after shutdown, want nil", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Serve did not return after context cancel")
	}
	if _, err := http.Get(base + "/healthz"); err == nil {
		t.Fatal("server still accepting connections after shutdown")
	}
}
