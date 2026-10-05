package netcheck

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	maxBodyRead    = 64 << 10 // never read more than this from an HTTP response
	maxRedirects   = 3
	userAgent      = "forgectl-netcheck"
	maxDetailAddrs = 5
)

// Dialer opens a connection. It is injectable so tests (and callers) control the network.
type Dialer func(ctx context.Context, network, address string) (net.Conn, error)

// Resolver looks up host names. *net.Resolver satisfies it.
type Resolver interface {
	LookupHost(ctx context.Context, host string) ([]string, error)
}

// Options configure Run. Zero values get safe defaults from DefaultOptions.
type Options struct {
	Concurrency  int           // checks in flight at once (default 8)
	Timeout      time.Duration // per-check timeout when the check sets none (default 5s)
	TotalTimeout time.Duration // wall-clock limit for the whole run (default 2m)
	Dialer       Dialer
	Resolver     Resolver
	Now          func() time.Time // used for certificate validity and expiry
}

// DefaultOptions returns the production options: real network, 8 workers, 5s per check, 2m total.
func DefaultOptions() Options {
	return Options{
		Concurrency:  8,
		Timeout:      5 * time.Second,
		TotalTimeout: 2 * time.Minute,
		Dialer:       (&net.Dialer{}).DialContext,
		Resolver:     net.DefaultResolver,
		Now:          time.Now,
	}
}

func (o *Options) fill() {
	d := DefaultOptions()
	if o.Concurrency < 1 {
		o.Concurrency = d.Concurrency
	}
	if o.Timeout <= 0 {
		o.Timeout = d.Timeout
	}
	if o.TotalTimeout <= 0 {
		o.TotalTimeout = d.TotalTimeout
	}
	if o.Dialer == nil {
		o.Dialer = d.Dialer
	}
	if o.Resolver == nil {
		o.Resolver = d.Resolver
	}
	if o.Now == nil {
		o.Now = d.Now
	}
}

// Statuses of a Result.
const (
	StatusPass = "pass"
	StatusFail = "fail"
	StatusWarn = "warn"
	StatusInfo = "info"
)

// Result is the outcome of one check.
type Result struct {
	ID           string `json:"id"`
	Kind         string `json:"kind"`
	Target       string `json:"target"`
	Expect       string `json:"expect"`
	Observed     string `json:"observed"`
	Status       string `json:"status"`
	Severity     string `json:"severity"`
	Detail       string `json:"detail"`
	DurationMS   int64  `json:"durationMs"`
	TLSVersion   string `json:"tlsVersion,omitempty"`
	CertNotAfter string `json:"certNotAfter,omitempty"`
	CertDaysLeft *int   `json:"certDaysLeft,omitempty"`
}

// Summary counts results by status.
type Summary struct {
	Pass int `json:"pass"`
	Fail int `json:"fail"`
	Warn int `json:"warn"`
	Info int `json:"info"`
}

// Report is the complete outcome of a run. Results are sorted by check id.
type Report struct {
	Summary Summary  `json:"summary"`
	Results []Result `json:"results"`
}

// Failed reports whether any check FAILed (the condition for exit code 1).
func (r Report) Failed() bool { return r.Summary.Fail > 0 }

// Run executes the checks with bounded concurrency and returns a report sorted by id. Checks that cannot start before
// the total timeout (or ctx cancellation) are reported as not run, with their severity applied as for any unmet check.
func Run(ctx context.Context, checks []Check, o Options) Report {
	o.fill()
	ctx, cancel := context.WithTimeout(ctx, o.TotalTimeout)
	defer cancel()

	results := make([]Result, len(checks))
	sem := make(chan struct{}, o.Concurrency)
	var wg sync.WaitGroup
	for i := range checks {
		i := i
		c := &checks[i]
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			results[i] = finish(c, outcome{observed: ObsError, detail: "not run: " + ctxReason(ctx)}, 0)
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			results[i] = runOne(ctx, c, &o)
		}()
	}
	wg.Wait()

	sort.SliceStable(results, func(a, b int) bool { return results[a].ID < results[b].ID })
	rep := Report{Results: results}
	for _, r := range results {
		switch r.Status {
		case StatusPass:
			rep.Summary.Pass++
		case StatusFail:
			rep.Summary.Fail++
		case StatusWarn:
			rep.Summary.Warn++
		case StatusInfo:
			rep.Summary.Info++
		}
	}
	return rep
}

func ctxReason(ctx context.Context) string {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return "total timeout exceeded"
	}
	return "interrupted"
}

// outcome is what a probe observed, before the expectation is applied.
type outcome struct {
	observed string // reachable | blocked | error
	detail   string
	warn     string // set when the probe passed but something deserves attention (certificate expiring soon)
	tls      string
	notAfter time.Time
	days     *int
}

func runOne(ctx context.Context, c *Check, o *Options) Result {
	start := time.Now()
	if ctx.Err() != nil {
		return finish(c, outcome{observed: ObsError, detail: "not run: " + ctxReason(ctx)}, 0)
	}
	timeout := c.timeout
	if timeout == 0 {
		timeout = o.Timeout
	}
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var out outcome
	switch c.Kind {
	case KindDNS:
		out = probeDNS(cctx, c, o)
	case KindTCP, KindEgressDeny:
		out = probeTCP(cctx, c, o)
	case KindTLS:
		out = probeTLS(cctx, c, o)
	case KindHTTP:
		out = probeHTTP(cctx, c, o, timeout)
	default:
		out = outcome{observed: ObsError, detail: "unknown check kind " + c.Kind}
	}
	return finish(c, out, time.Since(start))
}

// finish applies the expectation and severity to an outcome.
func finish(c *Check, out outcome, d time.Duration) Result {
	met := (c.Expect == Reachable && out.observed == Reachable) || (c.Expect == Blocked && out.observed == Blocked)
	r := Result{
		ID: c.ID, Kind: c.Kind, Target: c.Target, Expect: c.Expect, Observed: out.observed, Severity: c.Severity,
		Detail: out.detail, DurationMS: d.Milliseconds(), TLSVersion: out.tls, CertDaysLeft: out.days,
	}
	if !out.notAfter.IsZero() {
		r.CertNotAfter = out.notAfter.UTC().Format(time.RFC3339)
	}
	switch {
	case met && out.warn != "":
		r.Status = StatusWarn
		r.Detail = joinDetail(out.detail, out.warn)
	case met:
		r.Status = StatusPass
	default:
		switch c.Severity {
		case SevWarn:
			r.Status = StatusWarn
		case SevInfo:
			r.Status = StatusInfo
		default:
			r.Status = StatusFail
		}
		r.Detail = unmetDetail(c, out)
	}
	return r
}

func unmetDetail(c *Check, out outcome) string {
	if c.Kind == KindEgressDeny && out.observed == Reachable {
		return "LEAK: connection succeeded, so egress to " + c.Target + " is not blocked"
	}
	d := out.detail
	switch {
	case out.observed == Blocked && c.Expect == Reachable && c.hint != "":
		d = c.hint + ": " + d
	case out.observed == Reachable && c.Expect == Blocked:
		d = "expected blocked but it was reachable: " + d
	case strings.HasPrefix(d, "not run:"):
	default:
		d = "expected " + c.Expect + ", observed " + out.observed + ": " + d
	}
	return d
}

func joinDetail(a, b string) string {
	if a == "" {
		return b
	}
	return a + "; " + b
}

// ---------------------------------------------------------------- probes

func probeDNS(ctx context.Context, c *Check, o *Options) outcome {
	addrs, err := o.Resolver.LookupHost(ctx, c.host)
	if err != nil {
		var dnsErr *net.DNSError
		switch {
		case errors.As(err, &dnsErr) && dnsErr.IsNotFound:
			return outcome{observed: Blocked, detail: "no such host"}
		case isTimeout(ctx, err):
			return outcome{observed: Blocked, detail: "lookup timed out"}
		}
		return outcome{observed: Blocked, detail: "lookup failed: " + errText(err)}
	}
	if len(addrs) == 0 {
		return outcome{observed: Blocked, detail: "no addresses returned"}
	}
	sort.Strings(addrs)
	if c.cidr != nil {
		for _, a := range addrs {
			if ip := net.ParseIP(a); ip == nil || !c.cidr.Contains(ip) {
				return outcome{observed: ObsError, detail: fmt.Sprintf("resolved %s, outside %s", a, c.cidr)}
			}
		}
	}
	return outcome{observed: Reachable, detail: "resolved " + summarize(addrs)}
}

func probeTCP(ctx context.Context, c *Check, o *Options) outcome {
	conn, err := o.Dialer(ctx, "tcp", net.JoinHostPort(c.host, strconv.Itoa(c.port)))
	if err != nil {
		return outcome{observed: Blocked, detail: dialDetail(ctx, err)}
	}
	_ = conn.Close() // connect only: nothing is sent
	return outcome{observed: Reachable, detail: "connected"}
}

func probeTLS(ctx context.Context, c *Check, o *Options) outcome {
	conn, err := o.Dialer(ctx, "tcp", net.JoinHostPort(c.host, strconv.Itoa(c.port)))
	if err != nil {
		return outcome{observed: Blocked, detail: dialDetail(ctx, err)}
	}
	defer conn.Close()
	if dl, ok := ctx.Deadline(); ok {
		_ = conn.SetDeadline(dl)
	}
	name := c.ServerName
	if name == "" {
		name = c.host
	}
	tc := tls.Client(conn, &tls.Config{ServerName: name, MinVersion: c.minTLS, RootCAs: c.pool, Time: o.Now})
	if err := tc.HandshakeContext(ctx); err != nil {
		if isTimeout(ctx, err) {
			return outcome{observed: ObsError, detail: "connected, but the TLS handshake timed out"}
		}
		return outcome{observed: ObsError, detail: "TLS handshake failed: " + tlsErrText(err)}
	}
	st := tc.ConnectionState()
	out := outcome{observed: Reachable, tls: tlsVersionName(st.Version)}
	out.detail = "TLS " + out.tls + " verified for " + name
	annotateCert(&out, st.PeerCertificates, o.Now(), c.warnDays)
	return out
}

// annotateCert records the leaf certificate's expiry and flags certificates that expire within warnDays.
func annotateCert(out *outcome, certs []*x509.Certificate, now time.Time, warnDays int) {
	if len(certs) == 0 {
		return
	}
	na := certs[0].NotAfter
	days := int(na.Sub(now).Hours() / 24)
	out.notAfter, out.days = na, &days
	if days < warnDays {
		out.warn = fmt.Sprintf("certificate expires in %d day(s) (warn below %d)", days, warnDays)
	}
}

func probeHTTP(ctx context.Context, c *Check, o *Options, timeout time.Duration) outcome {
	var connected atomic.Bool
	dial := func(ctx context.Context, network, addr string) (net.Conn, error) {
		conn, err := o.Dialer(ctx, network, addr)
		if err == nil {
			connected.Store(true)
		}
		return conn, err
	}
	tr := &http.Transport{
		Proxy:                  nil, // measure the direct path; ignore HTTP(S)_PROXY
		DialContext:            dial,
		TLSClientConfig:        &tls.Config{MinVersion: c.minTLS, RootCAs: c.pool, Time: o.Now},
		DisableKeepAlives:      true,
		TLSHandshakeTimeout:    timeout,
		ResponseHeaderTimeout:  timeout,
		MaxResponseHeaderBytes: 64 << 10,
	}
	defer tr.CloseIdleConnections()

	var note string
	client := &http.Client{
		Transport: tr,
		Jar:       nil, // no cookies
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) > maxRedirects {
				note = fmt.Sprintf("stopped after %d redirects", maxRedirects)
				return http.ErrUseLastResponse
			}
			prev := via[len(via)-1]
			switch {
			case !strings.EqualFold(req.URL.Hostname(), prev.URL.Hostname()):
				note = "redirect to another host (" + req.URL.Host + ") not followed"
			case prev.URL.Scheme == "https" && req.URL.Scheme == "http" && !c.AllowHTTPRedirect:
				note = "redirect from https to http not followed (set allowHTTPRedirect to permit)"
			default:
				return nil
			}
			return http.ErrUseLastResponse
		},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.url.String(), nil)
	if err != nil {
		return outcome{observed: ObsError, detail: "bad request: " + errText(err)}
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := client.Do(req)
	if err != nil {
		if !connected.Load() {
			return outcome{observed: Blocked, detail: dialDetail(ctx, errors.Unwrap(err))}
		}
		if isTimeout(ctx, err) {
			return outcome{observed: ObsError, detail: "connected, but the response timed out"}
		}
		return outcome{observed: ObsError, detail: "request failed: " + tlsErrText(err)}
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxBodyRead))

	out := outcome{detail: "HTTP " + strconv.Itoa(resp.StatusCode)}
	if resp.TLS != nil {
		out.tls = tlsVersionName(resp.TLS.Version)
		annotateCert(&out, resp.TLS.PeerCertificates, o.Now(), c.warnDays)
	}
	if note != "" {
		out.detail += " (" + note + ")"
	}
	if resp.StatusCode != c.ExpectStatus {
		out.observed = ObsError
		out.detail += fmt.Sprintf(", want %d", c.ExpectStatus)
		if loc := resp.Header.Get("Location"); loc != "" && resp.StatusCode >= 300 && resp.StatusCode < 400 {
			out.detail += ", Location: " + truncate(loc, 120)
		}
		return out
	}
	out.observed = Reachable
	return out
}

// ---------------------------------------------------------------- helpers

func dialDetail(ctx context.Context, err error) string {
	switch {
	case err == nil:
		return "connection failed"
	case isTimeout(ctx, err):
		return "timed out (no response; typical of a firewall or NetworkPolicy drop)"
	case strings.Contains(strings.ToLower(err.Error()), "refused"):
		return "connection refused (host answered, port closed)"
	}
	return errText(err)
}

func isTimeout(ctx context.Context, err error) bool {
	var ne net.Error
	return errors.Is(err, context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded) ||
		(errors.As(err, &ne) && ne.Timeout())
}

// errText keeps error text short and strips nothing sensitive: netcheck never handles secrets.
func errText(err error) string { return truncate(err.Error(), 200) }

func tlsErrText(err error) string {
	var (
		hostErr    x509.HostnameError
		unknownErr x509.UnknownAuthorityError
		invalidErr x509.CertificateInvalidError
	)
	switch {
	case errors.As(err, &hostErr):
		return "certificate does not match the name: " + truncate(hostErr.Error(), 160)
	case errors.As(err, &invalidErr) && invalidErr.Reason == x509.Expired:
		return "certificate is expired or not yet valid"
	case errors.As(err, &unknownErr):
		return "certificate signed by an unknown authority (supply caFile?)"
	}
	return errText(err)
}

func tlsVersionName(v uint16) string {
	switch v {
	case tls.VersionTLS13:
		return "1.3"
	case tls.VersionTLS12:
		return "1.2"
	}
	return fmt.Sprintf("0x%04x", v)
}

func summarize(addrs []string) string {
	if len(addrs) <= maxDetailAddrs {
		return strings.Join(addrs, ", ")
	}
	return fmt.Sprintf("%s (+%d more)", strings.Join(addrs[:maxDetailAddrs], ", "), len(addrs)-maxDetailAddrs)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
