package netcheck

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func nowT() time.Time { return time.Now().Truncate(time.Second) }

func yamlPath(p string) string { return fmt.Sprintf("%q", filepath.ToSlash(p)) }

// ---------------------------------------------------------------- tcp and egress-deny

func TestTCP(t *testing.T) {
	open, closed := openPort(t), closedPort(t)
	cases := []struct {
		name   string
		extra  string
		port   int
		status string
		obs    string
	}{
		{"open port passes", "", open, StatusPass, Reachable},
		{"closed port fails", "", closed, StatusFail, Blocked},
		{"closed port, severity warn", ", severity: warn", closed, StatusWarn, Blocked},
		{"closed port, severity info", ", severity: info", closed, StatusInfo, Blocked},
		{"closed port, expect blocked", ", expect: blocked", closed, StatusPass, Blocked},
		{"open port, expect blocked", ", expect: blocked", open, StatusFail, Reachable},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := one(t, fmt.Sprintf("checks:\n  - {id: t, kind: tcp, target: \"127.0.0.1:%d\"%s}\n", c.port, c.extra), DefaultOptions())
			if r.Status != c.status || r.Observed != c.obs {
				t.Fatalf("status=%s observed=%s detail=%q; want %s/%s", r.Status, r.Observed, r.Detail, c.status, c.obs)
			}
		})
	}
}

func TestTCPRefusedDetail(t *testing.T) {
	r := one(t, fmt.Sprintf("checks:\n  - {id: t, kind: tcp, target: \"127.0.0.1:%d\"}\n", closedPort(t)), DefaultOptions())
	if !strings.Contains(r.Detail, "refused") {
		t.Errorf("detail should say refused, got %q", r.Detail)
	}
}

func TestEgressDeny(t *testing.T) {
	t.Run("closed is blocked", func(t *testing.T) {
		r := one(t, fmt.Sprintf("checks:\n  - {id: e, kind: egress-deny, target: \"127.0.0.1:%d\"}\n", closedPort(t)), DefaultOptions())
		if r.Status != StatusPass || r.Observed != Blocked {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("open is a leak", func(t *testing.T) {
		r := one(t, fmt.Sprintf("checks:\n  - {id: e, kind: egress-deny, target: \"127.0.0.1:%d\"}\n", openPort(t)), DefaultOptions())
		if r.Status != StatusFail || !strings.Contains(r.Detail, "LEAK") {
			t.Fatalf("a successful connection must be reported as a leak: %+v", r)
		}
	})
	t.Run("a drop (timeout) is blocked", func(t *testing.T) {
		o := DefaultOptions()
		o.Dialer = blockingDialer
		o.Timeout = 50 * time.Millisecond
		r := one(t, "checks:\n  - {id: e, kind: egress-deny, target: \"203.0.113.9:443\"}\n", o)
		if r.Status != StatusPass || !strings.Contains(r.Detail, "timed out") {
			t.Fatalf("%+v", r)
		}
	})
}

// ---------------------------------------------------------------- dns

type hangResolver struct{}

func (hangResolver) LookupHost(ctx context.Context, _ string) ([]string, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestDNS(t *testing.T) {
	res := fakeResolver{
		"svc.cluster.local": {"10.96.0.10"},
		"outside.example":   {"203.0.113.7"},
		"many.example":      {"10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.5", "10.0.0.6", "10.0.0.7"},
	}
	o := DefaultOptions()
	o.Resolver = res
	cases := []struct {
		name, extra, host, status, obs, detail string
	}{
		{"resolves", "", "svc.cluster.local", StatusPass, Reachable, "10.96.0.10"},
		{"nxdomain", "", "missing.example", StatusFail, Blocked, "no such host"},
		{"nxdomain expected blocked", ", expect: blocked", "missing.example", StatusPass, Blocked, "no such host"},
		{"inside CIDR", ", expectCIDR: 10.96.0.0/12", "svc.cluster.local", StatusPass, Reachable, ""},
		{"outside CIDR", ", expectCIDR: 10.96.0.0/12", "outside.example", StatusFail, ObsError, "outside 10.96.0.0/12"},
		{"many addresses summarised", "", "many.example", StatusPass, Reachable, "+2 more"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := one(t, fmt.Sprintf("checks:\n  - {id: d, kind: dns, target: %s%s}\n", c.host, c.extra), o)
			if r.Status != c.status || r.Observed != c.obs || !strings.Contains(r.Detail, c.detail) {
				t.Fatalf("%+v; want %s/%s containing %q", r, c.status, c.obs, c.detail)
			}
		})
	}
	t.Run("lookup timeout counts as blocked", func(t *testing.T) {
		o := DefaultOptions()
		o.Resolver = hangResolver{}
		o.Timeout = 50 * time.Millisecond
		r := one(t, "checks:\n  - {id: d, kind: dns, target: slow.example}\n", o)
		if r.Observed != Blocked || !strings.Contains(r.Detail, "timed out") {
			t.Fatalf("%+v", r)
		}
	})
}

// ---------------------------------------------------------------- tls

func tlsCheck(host string, port int, extra string) string {
	return fmt.Sprintf("checks:\n  - {id: t, kind: tls, target: \"%s:%d\"%s}\n", host, port, extra)
}

func TestTLS(t *testing.T) {
	T := nowT()
	ca := newCA(t, T.Add(-48*time.Hour), T.Add(24*time.Hour*365))
	caFile := yamlPath(ca.writeCA(t, t.TempDir(), "ca.pem"))
	good := ca.leaf(t, []string{"api.internal"}, loopback, T.Add(-time.Hour), T.Add(60*24*time.Hour))
	opts := func() Options { o := DefaultOptions(); o.Now = func() time.Time { return T }; return o }

	t.Run("valid certificate verified by IP SAN", func(t *testing.T) {
		h, p := startTLS(t, good, 0, false)
		r := one(t, tlsCheck(h, p, ", caFile: "+caFile), opts())
		if r.Status != StatusPass || r.TLSVersion == "" || r.CertDaysLeft == nil || *r.CertDaysLeft != 60 {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("serverName verified against the DNS SAN", func(t *testing.T) {
		h, p := startTLS(t, good, 0, false)
		r := one(t, tlsCheck(h, p, ", serverName: api.internal, caFile: "+caFile), opts())
		if r.Status != StatusPass {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("wrong SAN fails", func(t *testing.T) {
		h, p := startTLS(t, good, 0, false)
		r := one(t, tlsCheck(h, p, ", serverName: other.internal, caFile: "+caFile), opts())
		if r.Status != StatusFail || r.Observed != ObsError || !strings.Contains(r.Detail, "does not match") {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("certificate without a matching IP SAN fails", func(t *testing.T) {
		dnsOnly := ca.leaf(t, []string{"api.internal"}, nil, T.Add(-time.Hour), T.Add(60*24*time.Hour))
		h, p := startTLS(t, dnsOnly, 0, false)
		r := one(t, tlsCheck(h, p, ", caFile: "+caFile), opts())
		if r.Status != StatusFail || r.Observed != ObsError {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("expired certificate fails", func(t *testing.T) {
		expired := ca.leaf(t, nil, loopback, T.Add(-48*time.Hour), T.Add(-24*time.Hour))
		h, p := startTLS(t, expired, 0, false)
		r := one(t, tlsCheck(h, p, ", caFile: "+caFile), opts())
		if r.Status != StatusFail || !strings.Contains(r.Detail, "expired") {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("expiry is judged by the injected clock", func(t *testing.T) {
		h, p := startTLS(t, good, 0, false)
		o := opts()
		o.Now = func() time.Time { return T.Add(90 * 24 * time.Hour) } // after the 60 day certificate expires
		r := one(t, tlsCheck(h, p, ", caFile: "+caFile), o)
		if r.Status != StatusFail || !strings.Contains(r.Detail, "expired") {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("expiring soon warns but passes", func(t *testing.T) {
		soon := ca.leaf(t, nil, loopback, T.Add(-time.Hour), T.Add(5*24*time.Hour))
		h, p := startTLS(t, soon, 0, false)
		rep := Run(context.Background(), mustParse(t, tlsCheck(h, p, ", caFile: "+caFile)), opts())
		r := rep.Results[0]
		if r.Status != StatusWarn || r.Observed != Reachable || r.CertDaysLeft == nil || *r.CertDaysLeft != 5 ||
			!strings.Contains(r.Detail, "expires in 5 day") {
			t.Fatalf("%+v", r)
		}
		if rep.Failed() {
			t.Error("an expiry warning must not change the exit status")
		}
	})
	t.Run("warnExpiryDays lowers the threshold", func(t *testing.T) {
		soon := ca.leaf(t, nil, loopback, T.Add(-time.Hour), T.Add(5*24*time.Hour))
		h, p := startTLS(t, soon, 0, false)
		r := one(t, tlsCheck(h, p, ", warnExpiryDays: 3, caFile: "+caFile), opts())
		if r.Status != StatusPass {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("minimum TLS 1.3 against a TLS 1.2 server fails", func(t *testing.T) {
		h, p := startTLS(t, good, tls.VersionTLS12, false)
		r := one(t, tlsCheck(h, p, ", minTLSVersion: \"1.3\", caFile: "+caFile), opts())
		if r.Status != StatusFail || r.Observed != ObsError {
			t.Fatalf("%+v", r)
		}
		r = one(t, tlsCheck(h, p, ", minTLSVersion: \"1.2\", caFile: "+caFile), opts())
		if r.Status != StatusPass || r.TLSVersion != "1.2" {
			t.Fatalf("a 1.2 floor must accept a 1.2 server: %+v", r)
		}
	})
	t.Run("a certificate from an untrusted CA fails", func(t *testing.T) {
		h, p := startTLS(t, good, 0, false)
		r := one(t, tlsCheck(h, p, ""), opts()) // no caFile: system roots do not include the test CA
		if r.Status != StatusFail || r.Observed != ObsError {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("a stalled handshake times out as an error, not blocked", func(t *testing.T) {
		h, p := startTLS(t, good, 0, true)
		o := opts()
		o.Timeout = 150 * time.Millisecond
		r := one(t, tlsCheck(h, p, ", caFile: "+caFile), o)
		if r.Status != StatusFail || r.Observed != ObsError || !strings.Contains(r.Detail, "timed out") {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("closed port is blocked", func(t *testing.T) {
		r := one(t, tlsCheck("127.0.0.1", closedPort(t), ""), opts())
		if r.Status != StatusFail || r.Observed != Blocked {
			t.Fatalf("%+v", r)
		}
		r = one(t, tlsCheck("127.0.0.1", closedPort(t), ", expect: blocked"), opts())
		if r.Status != StatusPass {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("a reachable TLS server fails an expect-blocked check", func(t *testing.T) {
		h, p := startTLS(t, good, 0, false)
		r := one(t, tlsCheck(h, p, ", expect: blocked, caFile: "+caFile), opts())
		if r.Status != StatusFail || r.Observed != Reachable {
			t.Fatalf("%+v", r)
		}
	})
}

// ---------------------------------------------------------------- http

type recorder struct {
	mu      sync.Mutex
	headers []http.Header
}

func (r *recorder) add(h http.Header) {
	r.mu.Lock()
	r.headers = append(r.headers, h.Clone())
	r.mu.Unlock()
}

func TestHTTP(t *testing.T) {
	rec := &recorder{}
	mux := http.NewServeMux()
	mux.HandleFunc("/ok", func(w http.ResponseWriter, r *http.Request) { rec.add(r.Header); io.WriteString(w, "fine") })
	mux.HandleFunc("/missing", func(w http.ResponseWriter, r *http.Request) { http.NotFound(w, r) })
	mux.HandleFunc("/boom", func(w http.ResponseWriter, r *http.Request) { http.Error(w, "x", 500) })
	mux.HandleFunc("/to-other", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://other.invalid/secret", http.StatusFound)
	})
	mux.HandleFunc("/to-same", func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, "/ok", http.StatusFound) })
	mux.HandleFunc("/loop", func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, "/loop", http.StatusFound) })
	mux.HandleFunc("/hang", func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(3 * time.Second):
		}
	})
	mux.HandleFunc("/big", func(w http.ResponseWriter, r *http.Request) {
		chunk := make([]byte, 32<<10)
		for i := 0; i < 320; i++ { // 10 MiB
			if _, err := w.Write(chunk); err != nil {
				return
			}
		}
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	h, p := hostPort(t, srv.Listener.Addr().String())
	url := func(path string) string { return fmt.Sprintf("http://%s:%d%s", h, p, path) }
	doc := func(path, extra string) string {
		return fmt.Sprintf("checks:\n  - {id: h, kind: http, target: \"%s\"%s}\n", url(path), extra)
	}

	t.Run("200 passes", func(t *testing.T) {
		r := one(t, doc("/ok", ""), DefaultOptions())
		if r.Status != StatusPass || r.Detail != "HTTP 200" {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("expected non-200 status passes", func(t *testing.T) {
		if r := one(t, doc("/missing", ", expectStatus: 404"), DefaultOptions()); r.Status != StatusPass {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("unexpected status fails with both codes", func(t *testing.T) {
		r := one(t, doc("/boom", ""), DefaultOptions())
		if r.Status != StatusFail || r.Observed != ObsError || !strings.Contains(r.Detail, "HTTP 500, want 200") {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("redirect to another host is reported and never followed", func(t *testing.T) {
		var dialed []string
		var mu sync.Mutex
		o := DefaultOptions()
		o.Dialer = func(ctx context.Context, n, a string) (net.Conn, error) {
			mu.Lock()
			dialed = append(dialed, a)
			mu.Unlock()
			return (&net.Dialer{}).DialContext(ctx, n, a)
		}
		r := one(t, doc("/to-other", ""), o)
		if r.Status != StatusFail || !strings.Contains(r.Detail, "not followed") || !strings.Contains(r.Detail, "other.invalid") {
			t.Fatalf("%+v", r)
		}
		for _, a := range dialed {
			if strings.Contains(a, "other.invalid") {
				t.Fatalf("the redirect target was contacted: %v", dialed)
			}
		}
		// asserting the redirect itself is allowed, and still does not follow it
		if r := one(t, doc("/to-other", ", expectStatus: 302"), o); r.Status != StatusPass {
			t.Fatalf("expectStatus 302 should pass at the first response: %+v", r)
		}
	})
	t.Run("a same-host redirect is followed", func(t *testing.T) {
		if r := one(t, doc("/to-same", ""), DefaultOptions()); r.Status != StatusPass {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("a redirect loop stops after three hops", func(t *testing.T) {
		r := one(t, doc("/loop", ""), DefaultOptions())
		if r.Status != StatusFail || !strings.Contains(r.Detail, "stopped after 3 redirects") {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("no credentials are sent", func(t *testing.T) {
		rec.mu.Lock()
		rec.headers = nil
		rec.mu.Unlock()
		one(t, doc("/ok", ""), DefaultOptions())
		rec.mu.Lock()
		defer rec.mu.Unlock()
		if len(rec.headers) != 1 {
			t.Fatalf("expected one request, got %d", len(rec.headers))
		}
		hd := rec.headers[0]
		for _, k := range []string{"Authorization", "Cookie", "Proxy-Authorization"} {
			if hd.Get(k) != "" {
				t.Errorf("%s header was sent", k)
			}
		}
		if hd.Get("User-Agent") != userAgent {
			t.Errorf("User-Agent = %q", hd.Get("User-Agent"))
		}
	})
	t.Run("the response body read is capped", func(t *testing.T) {
		var read atomic.Int64
		o := DefaultOptions()
		o.Dialer = func(ctx context.Context, n, a string) (net.Conn, error) {
			c, err := (&net.Dialer{}).DialContext(ctx, n, a)
			if err != nil {
				return nil, err
			}
			return &countingConn{Conn: c, n: &read}, nil
		}
		if r := one(t, doc("/big", ""), o); r.Status != StatusPass {
			t.Fatalf("%+v", r)
		}
		if n := read.Load(); n > 512<<10 {
			t.Errorf("read %d bytes of a 10 MiB body; the cap is %d plus socket buffering", n, maxBodyRead)
		}
	})
	t.Run("a hung response times out as an error", func(t *testing.T) {
		o := DefaultOptions()
		o.Timeout = 150 * time.Millisecond
		r := one(t, doc("/hang", ""), o)
		if r.Status != StatusFail || r.Observed != ObsError || !strings.Contains(r.Detail, "timed out") {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("closed port is blocked", func(t *testing.T) {
		r := one(t, fmt.Sprintf("checks:\n  - {id: h, kind: http, target: \"http://127.0.0.1:%d/\"}\n", closedPort(t)), DefaultOptions())
		if r.Status != StatusFail || r.Observed != Blocked {
			t.Fatalf("%+v", r)
		}
	})
}

type countingConn struct {
	net.Conn
	n *atomic.Int64
}

func (c *countingConn) Read(b []byte) (int, error) {
	n, err := c.Conn.Read(b)
	c.n.Add(int64(n))
	return n, err
}

func TestHTTPS(t *testing.T) {
	T := nowT()
	ca := newCA(t, T.Add(-48*time.Hour), T.Add(24*time.Hour*365))
	caFile := yamlPath(ca.writeCA(t, t.TempDir(), "ca.pem"))
	cert := ca.leaf(t, []string{"localhost"}, loopback, T.Add(-time.Hour), T.Add(60*24*time.Hour))

	plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "plain") }))
	t.Cleanup(plain.Close)
	secure := startHTTPS(t, cert, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/downgrade" {
			http.Redirect(w, r, plain.URL+"/", http.StatusFound)
			return
		}
		io.WriteString(w, "secure")
	}))
	h, p := hostPort(t, secure.Listener.Addr().String())
	doc := func(path, extra string) string {
		return fmt.Sprintf("checks:\n  - {id: h, kind: http, target: \"https://%s:%d%s\"%s}\n", h, p, path, extra)
	}

	t.Run("verified HTTPS passes and reports the certificate", func(t *testing.T) {
		r := one(t, doc("/", ", caFile: "+caFile), DefaultOptions())
		if r.Status != StatusPass || r.TLSVersion == "" || r.CertDaysLeft == nil {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("an untrusted CA fails as an error", func(t *testing.T) {
		r := one(t, doc("/", ""), DefaultOptions())
		if r.Status != StatusFail || r.Observed != ObsError {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("https to http redirect is not followed by default", func(t *testing.T) {
		r := one(t, doc("/downgrade", ", caFile: "+caFile), DefaultOptions())
		if r.Status != StatusFail || !strings.Contains(r.Detail, "https to http") {
			t.Fatalf("%+v", r)
		}
	})
	t.Run("https to http redirect is followed when allowed", func(t *testing.T) {
		r := one(t, doc("/downgrade", ", allowHTTPRedirect: true, caFile: "+caFile), DefaultOptions())
		if r.Status != StatusPass {
			t.Fatalf("%+v", r)
		}
	})
}

// ---------------------------------------------------------------- run behaviour

func TestConcurrencyIsBounded(t *testing.T) {
	var cur, peak atomic.Int64
	o := DefaultOptions()
	o.Concurrency = 3
	o.Dialer = func(ctx context.Context, _, _ string) (net.Conn, error) {
		n := cur.Add(1)
		for {
			p := peak.Load()
			if n <= p || peak.CompareAndSwap(p, n) {
				break
			}
		}
		defer cur.Add(-1)
		select {
		case <-time.After(40 * time.Millisecond):
		case <-ctx.Done():
		}
		return nil, errors.New("connection refused")
	}
	var b strings.Builder
	b.WriteString("checks:\n")
	for i := 0; i < 12; i++ {
		fmt.Fprintf(&b, "  - {id: c%02d, kind: tcp, target: \"h.example:%d\", expect: blocked}\n", i, 1000+i)
	}
	rep := Run(context.Background(), mustParse(t, b.String()), o)
	if rep.Summary.Pass != 12 {
		t.Fatalf("summary %+v", rep.Summary)
	}
	if got := peak.Load(); got != 3 {
		t.Fatalf("peak concurrency %d, want exactly the limit 3", got)
	}
}

func TestPerCheckTimeout(t *testing.T) {
	o := DefaultOptions()
	o.Dialer = blockingDialer
	o.Timeout = 50 * time.Millisecond
	start := time.Now()
	r := one(t, "checks:\n  - {id: t, kind: tcp, target: \"203.0.113.1:80\"}\n", o)
	if r.Observed != Blocked || !strings.Contains(r.Detail, "timed out") {
		t.Fatalf("%+v", r)
	}
	if time.Since(start) > 3*time.Second {
		t.Fatal("the check did not honour its timeout")
	}
}

func TestCheckTimeoutOverridesDefault(t *testing.T) {
	o := DefaultOptions()
	o.Dialer = blockingDialer
	o.Timeout = time.Minute // would hang the test if the per-check value were ignored
	start := time.Now()
	r := one(t, "checks:\n  - {id: t, kind: tcp, target: \"203.0.113.1:80\", timeout: 50ms}\n", o)
	if r.Observed != Blocked || time.Since(start) > 3*time.Second {
		t.Fatalf("%+v after %s", r, time.Since(start))
	}
}

func TestTotalTimeoutStopsTheRun(t *testing.T) {
	o := DefaultOptions()
	o.Concurrency = 1
	o.Dialer = blockingDialer
	o.Timeout = time.Minute
	o.TotalTimeout = 150 * time.Millisecond
	var b strings.Builder
	b.WriteString("checks:\n")
	for i := 0; i < 5; i++ {
		fmt.Fprintf(&b, "  - {id: c%d, kind: tcp, target: \"203.0.113.1:%d\"}\n", i, 80+i)
	}
	start := time.Now()
	rep := Run(context.Background(), mustParse(t, b.String()), o)
	if time.Since(start) > 3*time.Second {
		t.Fatal("total timeout was not enforced")
	}
	notRun := 0
	for _, r := range rep.Results {
		if r.Status == StatusPass {
			t.Errorf("%s passed", r.ID)
		}
		if strings.HasPrefix(r.Detail, "not run:") {
			notRun++
		}
	}
	if notRun < 4 || !rep.Failed() {
		t.Fatalf("expected at least 4 checks not run and a failed report: %+v", rep)
	}
}

func TestCancelledContextRunsNothing(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	rep := Run(ctx, mustParse(t, "checks:\n  - {id: a, kind: tcp, target: \"127.0.0.1:1\"}\n"), DefaultOptions())
	if rep.Results[0].Status != StatusFail || !strings.Contains(rep.Results[0].Detail, "not run") {
		t.Fatalf("%+v", rep.Results[0])
	}
}

func TestOutputIsSortedAndStable(t *testing.T) {
	open := openPort(t)
	doc := fmt.Sprintf(`checks:
  - {id: zulu, kind: tcp, target: "127.0.0.1:%d"}
  - {id: alpha, kind: tcp, target: "127.0.0.1:%d"}
  - {id: mike, kind: tcp, target: "127.0.0.1:%d", expect: blocked}
  - {id: bravo, kind: egress-deny, target: "127.0.0.1:%d"}
`, open, open, open, closedPort(t))
	checks := mustParse(t, doc)
	o := DefaultOptions()
	strip := func(r Report) string {
		for i := range r.Results {
			r.Results[i].DurationMS = 0
		}
		b, err := r.JSON()
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	first := Run(context.Background(), checks, o)
	ids := []string{}
	for _, r := range first.Results {
		ids = append(ids, r.ID)
	}
	if strings.Join(ids, ",") != "alpha,bravo,mike,zulu" {
		t.Fatalf("results are not sorted by id: %v", ids)
	}
	for i := 0; i < 5; i++ {
		if got := strip(Run(context.Background(), checks, o)); got != strip(first) {
			t.Fatalf("run %d differs:\n%s\nvs\n%s", i, got, strip(first))
		}
	}
}

func TestReportFailedFollowsSeverity(t *testing.T) {
	closed := closedPort(t)
	mk := func(sev string) Report {
		return Run(context.Background(),
			mustParse(t, fmt.Sprintf("checks:\n  - {id: a, kind: tcp, target: \"127.0.0.1:%d\", severity: %s}\n", closed, sev)),
			DefaultOptions())
	}
	if mk(SevError).Failed() != true {
		t.Error("an unmet error-severity check must fail the report")
	}
	if mk(SevWarn).Failed() || mk(SevInfo).Failed() {
		t.Error("unmet warn/info checks must not fail the report")
	}
}

func TestRenderAndJSON(t *testing.T) {
	rep := Run(context.Background(), mustParse(t, fmt.Sprintf(
		"checks:\n  - {id: up, kind: tcp, target: \"127.0.0.1:%d\"}\n  - {id: down, kind: tcp, target: \"127.0.0.1:%d\"}\n",
		openPort(t), closedPort(t))), DefaultOptions())
	text := Render(rep)
	for _, want := range []string{"STATUS", "PASS", "FAIL", "up", "down", "1 pass, 0 warn, 0 info, 1 fail"} {
		if !strings.Contains(text, want) {
			t.Errorf("table missing %q:\n%s", want, text)
		}
	}
	b, err := rep.JSON()
	if err != nil {
		t.Fatal(err)
	}
	var back Report
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back.Summary.Pass != 1 || back.Summary.Fail != 1 || len(back.Results) != 2 || back.Results[0].ID != "down" {
		t.Fatalf("JSON round trip: %+v", back)
	}
}

// ---------------------------------------------------------------- local preset

func TestLocalPreset(t *testing.T) {
	cs := LocalPreset()
	ports := map[string]bool{}
	for _, c := range cs {
		ports[c.Target] = true
		if c.Severity != SevInfo || c.Expect != Reachable {
			t.Errorf("%s: preset checks must be info-severity and expect reachable: %+v", c.ID, c)
		}
	}
	for _, want := range []string{"127.0.0.1:8000", "127.0.0.1:7070", "127.0.0.1:9090", "127.0.0.1:3000", "127.0.0.1:9835"} {
		if !ports[want] {
			t.Errorf("preset is missing %s", want)
		}
	}

	t.Run("nothing listening is info, never a failure", func(t *testing.T) {
		o := DefaultOptions()
		o.Dialer = func(context.Context, string, string) (net.Conn, error) { return nil, errors.New("connection refused") }
		rep := Run(context.Background(), LocalPreset(), o)
		if rep.Failed() || rep.Summary.Info != len(cs) {
			t.Fatalf("summary %+v", rep.Summary)
		}
		for _, r := range rep.Results {
			if !strings.Contains(r.Detail, "not listening") {
				t.Errorf("%s detail %q should say not listening", r.ID, r.Detail)
			}
		}
	})
	t.Run("a listening service passes", func(t *testing.T) {
		o := DefaultOptions()
		o.Dialer = func(_ context.Context, _, addr string) (net.Conn, error) {
			if addr == "127.0.0.1:8000" {
				a, b := net.Pipe()
				b.Close()
				return a, nil
			}
			return nil, errors.New("connection refused")
		}
		rep := Run(context.Background(), LocalPreset(), o)
		got := map[string]string{}
		for _, r := range rep.Results {
			got[r.ID] = r.Status
		}
		if got["local-api"] != StatusPass || got["local-agent"] != StatusInfo {
			t.Fatalf("%v", got)
		}
	})
}
