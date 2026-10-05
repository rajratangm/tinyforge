// Package netcheck runs a declarative reachability matrix (`forgectl net check`).
//
// It is a connectivity checker, NOT a scanner: every target is one explicit host:port (or URL, or hostname for DNS).
// CIDR ranges, port ranges, wildcards and comma lists are rejected when the file is loaded, and the number of checks
// is capped. It sends no credentials (URLs with user info are rejected, no cookies, no auth headers) and ignores
// proxy environment variables, so what it reports is the direct network path from this machine or pod.
//
// Checks file (YAML or JSON; unknown fields are rejected):
//
//	apiVersion: tinyforge.dev/v1alpha1     # optional; if set it must be exactly this
//	kind: NetCheck                         # optional; if set it must be exactly this
//	checks:
//	  - id: api-tls                        # required, unique, [a-z0-9][a-z0-9._-]*
//	    kind: tls                          # dns | tcp | tls | http | egress-deny
//	    target: "api.internal:8000"        # tls/tcp/egress-deny: host:port. http: http(s):// URL. dns: hostname
//	    expect: reachable                  # reachable (default) | blocked. egress-deny is always blocked
//	    severity: error                    # error (default) | warn | info: how an UNMET expectation is reported
//	    timeout: 5s                        # optional per-check timeout, 1ms..2m (default: --timeout)
//
//	    # dns only
//	    expectCIDR: 10.96.0.0/12           # EVERY resolved address (IPv4 and IPv6) must be inside this CIDR; a name that
//	                                       # also has an AAAA record fails an IPv4-only CIDR (localhost resolves to ::1)
//	    # tls and http
//	    minTLSVersion: "1.2"               # "1.2" (default) or "1.3". 1.0 and 1.1 are not accepted
//	    caFile: ca.pem                     # PEM bundle to trust instead of the system roots (relative to this file)
//	    # tls only
//	    serverName: api.internal           # name to verify the certificate against (default: the target host)
//	    warnExpiryDays: 14                 # certificate expiring sooner than this is reported as WARN (default 14)
//	    # http only
//	    expectStatus: 200                  # default 200
//	    allowHTTPRedirect: false           # permit a same-host redirect from https to http
//
// Semantics. "Observed" is one of reachable, blocked or error:
//
//	blocked   the connection could not be made (refused, unreachable, timed out) or the name did not resolve
//	reachable the check succeeded completely (connected / handshake verified / expected HTTP status / name resolved in CIDR)
//	error     something answered but the check failed (bad certificate, wrong status, resolved outside the CIDR, ...)
//
// An expectation is met when expect=reachable and observed=reachable, or expect=blocked and observed=blocked. An unmet
// expectation is a FAIL for severity error, a WARN for warn and INFO for info. The exit code is 1 only when some
// check FAILed. A certificate that verifies but expires soon is a WARN and does not change the exit code.
//
// For egress-deny, "blocked" includes "connection refused", which means the packet reached a host that answered with
// a reset: the destination was reachable but the port was closed. A NetworkPolicy or firewall drop normally shows up
// as a timeout. The detail text says which one was seen.
//
// Planned, not implemented: path-MTU probing, bandwidth (iperf3/nccl-tests), UDP checks.
package netcheck

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const (
	APIVersion = "tinyforge.dev/v1alpha1"
	FileKind   = "NetCheck"

	MaxChecks      = 256
	maxTimeout     = 2 * time.Minute
	defaultWarnDay = 14
)

// Check kinds.
const (
	KindDNS        = "dns"
	KindTCP        = "tcp"
	KindTLS        = "tls"
	KindHTTP       = "http"
	KindEgressDeny = "egress-deny"
)

// Expectations, severities and observations.
const (
	Reachable = "reachable"
	Blocked   = "blocked"
	ObsError  = "error"

	SevError = "error"
	SevWarn  = "warn"
	SevInfo  = "info"
)

// File is the on-disk document.
type File struct {
	APIVersion string  `yaml:"apiVersion,omitempty"`
	Kind       string  `yaml:"kind,omitempty"`
	Checks     []Check `yaml:"checks"`
}

// Check is one entry of a checks file. Fields below the resolved marker are filled in by validation.
type Check struct {
	ID       string `yaml:"id"`
	Kind     string `yaml:"kind"`
	Target   string `yaml:"target"`
	Expect   string `yaml:"expect,omitempty"`
	Severity string `yaml:"severity,omitempty"`
	Timeout  string `yaml:"timeout,omitempty"`

	ExpectCIDR        string `yaml:"expectCIDR,omitempty"`
	MinTLSVersion     string `yaml:"minTLSVersion,omitempty"`
	CAFile            string `yaml:"caFile,omitempty"`
	ServerName        string `yaml:"serverName,omitempty"`
	WarnExpiryDays    *int   `yaml:"warnExpiryDays,omitempty"`
	ExpectStatus      int    `yaml:"expectStatus,omitempty"`
	AllowHTTPRedirect bool   `yaml:"allowHTTPRedirect,omitempty"`

	// resolved by validate(); never read from the file
	hint     string // prefix for "blocked" details (used by the local preset)
	host     string
	port     int
	url      *url.URL
	timeout  time.Duration
	minTLS   uint16
	cidr     *net.IPNet
	pool     *x509.CertPool
	warnDays int
	blocked  bool // expectation is "blocked"
}

var (
	idRe       = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,62}$`)
	hostnameRe = regexp.MustCompile(`^[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$`)
)

// Load reads, strictly parses and validates a checks file. Relative caFile paths resolve against the file's directory.
func Load(path string) ([]Check, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(raw, filepath.Dir(path))
}

// Parse strictly parses and validates a checks document held in memory.
func Parse(raw []byte, baseDir string) ([]Check, error) {
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	var f File
	if err := dec.Decode(&f); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, errors.New("checks file is empty")
		}
		return nil, fmt.Errorf("parse: %w", err)
	}
	var extra File
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		return nil, errors.New("parse: exactly one YAML document is allowed")
	}
	if f.APIVersion != "" && f.APIVersion != APIVersion {
		return nil, fmt.Errorf("apiVersion %q is not supported (want %q)", f.APIVersion, APIVersion)
	}
	if f.Kind != "" && f.Kind != FileKind {
		return nil, fmt.Errorf("kind %q is not supported (want %q)", f.Kind, FileKind)
	}
	if len(f.Checks) == 0 {
		return nil, errors.New("checks: at least one check is required")
	}
	if len(f.Checks) > MaxChecks {
		return nil, fmt.Errorf("checks: %d checks exceed the limit of %d (this is a connectivity checker, not a scanner)",
			len(f.Checks), MaxChecks)
	}
	var errs []error
	seen := map[string]bool{}
	for i := range f.Checks {
		c := &f.Checks[i]
		if err := c.resolve(baseDir); err != nil {
			errs = append(errs, fmt.Errorf("check %s: %w", label(c, i), err))
			continue
		}
		if seen[c.ID] {
			errs = append(errs, fmt.Errorf("check %s: duplicate id", label(c, i)))
		}
		seen[c.ID] = true
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return f.Checks, nil
}

func label(c *Check, i int) string {
	if c.ID != "" {
		return strconv.Quote(c.ID)
	}
	return fmt.Sprintf("#%d", i+1)
}

// kindFields lists the optional fields each kind accepts; anything else set on a check is rejected.
var kindFields = map[string][]string{
	KindDNS:        {"expectCIDR"},
	KindTCP:        {},
	KindTLS:        {"minTLSVersion", "caFile", "serverName", "warnExpiryDays"},
	KindHTTP:       {"minTLSVersion", "caFile", "expectStatus", "allowHTTPRedirect"},
	KindEgressDeny: {},
}

func (c *Check) setFields() []string {
	var s []string
	add := func(name string, set bool) {
		if set {
			s = append(s, name)
		}
	}
	add("expectCIDR", c.ExpectCIDR != "")
	add("minTLSVersion", c.MinTLSVersion != "")
	add("caFile", c.CAFile != "")
	add("serverName", c.ServerName != "")
	add("warnExpiryDays", c.WarnExpiryDays != nil)
	add("expectStatus", c.ExpectStatus != 0)
	add("allowHTTPRedirect", c.AllowHTTPRedirect)
	return s
}

func (c *Check) resolve(baseDir string) error {
	if !idRe.MatchString(c.ID) {
		return fmt.Errorf("id %q must match %s", c.ID, idRe)
	}
	allowed, ok := kindFields[c.Kind]
	if !ok {
		return fmt.Errorf("kind %q is not one of dns, tcp, tls, http, egress-deny", c.Kind)
	}
	for _, f := range c.setFields() {
		if !contains(allowed, f) {
			return fmt.Errorf("field %s does not apply to kind %s", f, c.Kind)
		}
	}

	switch c.Expect {
	case "":
		c.Expect = Reachable
		if c.Kind == KindEgressDeny {
			c.Expect = Blocked
		}
	case Reachable, Blocked:
	default:
		return fmt.Errorf("expect %q must be reachable or blocked", c.Expect)
	}
	if c.Kind == KindEgressDeny && c.Expect != Blocked {
		return errors.New("egress-deny always expects blocked; use kind tcp to assert something is reachable")
	}
	c.blocked = c.Expect == Blocked

	switch c.Severity {
	case "":
		c.Severity = SevError
	case SevError, SevWarn, SevInfo:
	default:
		return fmt.Errorf("severity %q must be error, warn or info", c.Severity)
	}

	c.timeout = 0
	if c.Timeout != "" {
		d, err := time.ParseDuration(c.Timeout)
		if err != nil || d < time.Millisecond || d > maxTimeout {
			return fmt.Errorf("timeout %q must be a duration between 1ms and %s", c.Timeout, maxTimeout)
		}
		c.timeout = d
	}

	var err error
	switch c.Kind {
	case KindDNS:
		err = c.resolveDNS()
	case KindTCP, KindEgressDeny:
		c.host, c.port, err = parseHostPort(c.Target)
	case KindTLS:
		if c.host, c.port, err = parseHostPort(c.Target); err == nil {
			err = c.resolveTLS(baseDir)
		}
	case KindHTTP:
		if err = c.resolveHTTP(); err == nil {
			err = c.resolveTLS(baseDir)
		}
	}
	return err
}

func (c *Check) resolveDNS() error {
	if err := checkHostname(c.Target, false); err != nil {
		return fmt.Errorf("target %q: %w", c.Target, err)
	}
	c.host = c.Target
	if c.ExpectCIDR != "" {
		_, n, err := net.ParseCIDR(c.ExpectCIDR)
		if err != nil {
			return fmt.Errorf("expectCIDR %q is not a valid CIDR", c.ExpectCIDR)
		}
		c.cidr = n
	}
	return nil
}

func (c *Check) resolveHTTP() error {
	u, err := url.Parse(c.Target)
	if err != nil || u.Host == "" {
		return fmt.Errorf("target %q must be an http:// or https:// URL", c.Target)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("target %q: scheme must be http or https", c.Target)
	}
	if u.User != nil {
		return errors.New("target URL must not contain credentials (netcheck sends none)")
	}
	if strings.ContainsAny(u.Host, "/,* ") || strings.Contains(u.Hostname(), "..") {
		return fmt.Errorf("target %q: a single explicit host is required", c.Target)
	}
	if err := checkHostname(u.Hostname(), true); err != nil {
		return fmt.Errorf("target %q: %w", c.Target, err)
	}
	port := 80
	if u.Scheme == "https" {
		port = 443
	}
	if p := u.Port(); p != "" {
		if port, err = parsePort(p); err != nil {
			return fmt.Errorf("target %q: %w", c.Target, err)
		}
	}
	c.url, c.host, c.port = u, u.Hostname(), port
	if c.ExpectStatus == 0 {
		c.ExpectStatus = 200
	}
	if c.ExpectStatus < 100 || c.ExpectStatus > 599 {
		return fmt.Errorf("expectStatus %d is not an HTTP status code", c.ExpectStatus)
	}
	return nil
}

// resolveTLS handles the TLS options shared by the tls and http kinds.
func (c *Check) resolveTLS(baseDir string) error {
	switch c.MinTLSVersion {
	case "", "1.2":
		c.minTLS = tls.VersionTLS12
	case "1.3":
		c.minTLS = tls.VersionTLS13
	default:
		return fmt.Errorf("minTLSVersion %q must be \"1.2\" or \"1.3\"", c.MinTLSVersion)
	}
	c.warnDays = defaultWarnDay
	if c.WarnExpiryDays != nil {
		if *c.WarnExpiryDays < 0 || *c.WarnExpiryDays > 3650 {
			return fmt.Errorf("warnExpiryDays %d must be between 0 and 3650", *c.WarnExpiryDays)
		}
		c.warnDays = *c.WarnExpiryDays
	}
	if c.ServerName != "" {
		if err := checkHostname(c.ServerName, true); err != nil {
			return fmt.Errorf("serverName %q: %w", c.ServerName, err)
		}
	}
	if c.CAFile != "" {
		p := c.CAFile
		if !filepath.IsAbs(p) {
			p = filepath.Join(baseDir, p)
		}
		pem, err := os.ReadFile(p)
		if err != nil {
			return fmt.Errorf("caFile: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return fmt.Errorf("caFile %s contains no PEM certificates", c.CAFile)
		}
		c.pool = pool
	}
	return nil
}

// parseHostPort accepts exactly one explicit host:port. Ranges, CIDRs, wildcards and lists are refused.
func parseHostPort(target string) (string, int, error) {
	if strings.ContainsAny(target, "/,* \t") {
		return "", 0, fmt.Errorf("target %q must be a single host:port (no CIDR, wildcard or list)", target)
	}
	host, p, err := net.SplitHostPort(target)
	if err != nil {
		return "", 0, fmt.Errorf("target %q must be host:port", target)
	}
	port, err := parsePort(p)
	if err != nil {
		return "", 0, fmt.Errorf("target %q: %w (port ranges are not allowed)", target, err)
	}
	if err := checkHostname(host, true); err != nil {
		return "", 0, fmt.Errorf("target %q: %w", target, err)
	}
	return host, port, nil
}

func parsePort(s string) (int, error) {
	n, err := strconv.Atoi(s)
	if err != nil || n < 1 || n > 65535 {
		return 0, fmt.Errorf("port %q must be a number from 1 to 65535", s)
	}
	return n, nil
}

// checkHostname validates a DNS name or, when allowIP is set, an IP literal.
func checkHostname(h string, allowIP bool) error {
	if h == "" {
		return errors.New("host is empty")
	}
	if ip := net.ParseIP(h); ip != nil {
		if !allowIP {
			return errors.New("an IP address cannot be used for a dns check; give a hostname")
		}
		return nil
	}
	if strings.ContainsAny(h, "/%") || !hostnameRe.MatchString(h) || strings.Contains(h, "..") {
		return fmt.Errorf("%q is not a valid hostname or IP address", h)
	}
	return nil
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}
