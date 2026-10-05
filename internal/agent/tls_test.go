package agent

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"io"
	"log"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// ---- in-memory PKI -------------------------------------------------------------------------------------------

type pki struct {
	cert *x509.Certificate
	key  *ecdsa.PrivateKey
	pem  []byte
}

func newCA(t *testing.T, cn string) *pki {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: cn},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(24 * time.Hour),
		IsCA: true, BasicConstraintsValid: true, KeyUsage: x509.KeyUsageCertSign | x509.KeyUsageDigitalSignature,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}
	return &pki{cert: cert, key: key, pem: pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})}
}

// issue signs a fresh key pair. server=true gives a serverAuth leaf valid for 127.0.0.1 and localhost.
func (p *pki) issue(t *testing.T, cn string, serial int64, server bool) (certPEM, keyPEM []byte) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(serial), Subject: pkix.Name{CommonName: cn},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(24 * time.Hour),
		KeyUsage: x509.KeyUsageDigitalSignature,
	}
	if server {
		tmpl.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}
		tmpl.DNSNames = []string{"localhost"}
		tmpl.IPAddresses = []net.IP{net.ParseIP("127.0.0.1")}
	} else {
		tmpl.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, p.cert, &key.PublicKey, p.key)
	if err != nil {
		t.Fatal(err)
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}),
		pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})
}

func (p *pki) pool() *x509.CertPool {
	pool := x509.NewCertPool()
	pool.AddCert(p.cert)
	return pool
}

func writeFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

type syncBuf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuf) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuf) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// ---- test environment ----------------------------------------------------------------------------------------

type tlsEnv struct {
	s     *Server
	base  string // https://127.0.0.1:port
	dir   string
	ca    *pki
	cert  string // server cert path
	key   string // server key path
	caPEM string // client CA bundle path
	logs  *syncBuf
}

// startTLS writes a server certificate and a client CA bundle to a temp dir, starts the agent on a loopback
// port and returns once it is serving. mut may adjust the Config (add the client CA, token, intervals...).
func startTLS(t *testing.T, mut func(*Config, *tlsEnv)) *tlsEnv {
	t.Helper()
	e := &tlsEnv{dir: t.TempDir(), ca: newCA(t, "test-ca"), logs: &syncBuf{}}
	e.cert, e.key, e.caPEM = filepath.Join(e.dir, "server.pem"), filepath.Join(e.dir, "server.key"), filepath.Join(e.dir, "ca.pem")
	certPEM, keyPEM := e.ca.issue(t, "agent", 100, true)
	writeFile(t, e.cert, certPEM)
	writeFile(t, e.key, keyPEM)
	writeFile(t, e.caPEM, e.ca.pem)

	s, _ := newTestServer(t, func(c *Config) {
		c.Log = log.New(e.logs, "", 0)
		c.TLSCertFile, c.TLSKeyFile = e.cert, e.key
		c.RefreshEvery = 20 * time.Millisecond
		if mut != nil {
			mut(c, e)
		}
	})
	e.s = s
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	e.base = "https://" + ln.Addr().String()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- s.Serve(ctx, ln) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			t.Error("agent did not shut down")
		}
	})

	// Ready = the first doctor report exists. Use a plain TCP+TLS client that trusts the CA; under mTLS the caller
	// supplies the client certificate through mut, so poll with the matching client when one is configured.
	return e
}

func (e *tlsEnv) client(t *testing.T, trust *x509.CertPool, clientCert *tls.Certificate, mut func(*tls.Config)) *http.Client {
	t.Helper()
	cfg := &tls.Config{RootCAs: trust}
	if clientCert != nil {
		cfg.Certificates = []tls.Certificate{*clientCert}
	}
	if mut != nil {
		mut(cfg)
	}
	return &http.Client{
		Timeout:   5 * time.Second,
		Transport: &http.Transport{TLSClientConfig: cfg, DisableKeepAlives: true},
	}
}

func (e *tlsEnv) clientCert(t *testing.T, ca *pki, cn string) *tls.Certificate {
	t.Helper()
	c, k := ca.issue(t, cn, 7, false)
	pair, err := tls.X509KeyPair(c, k)
	if err != nil {
		t.Fatal(err)
	}
	return &pair
}

func (e *tlsEnv) waitReady(t *testing.T, c *http.Client) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		resp, err := c.Get(e.base + "/readyz")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("agent never became ready (last err: %v)", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func peerSerial(t *testing.T, c *http.Client, url string) int64 {
	t.Helper()
	resp, err := c.Get(url + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.TLS == nil || len(resp.TLS.PeerCertificates) == 0 {
		t.Fatal("no TLS peer certificate on the response")
	}
	return resp.TLS.PeerCertificates[0].SerialNumber.Int64()
}

// ---- TLS -----------------------------------------------------------------------------------------------------

func TestTLSHandshakeWithCA(t *testing.T) {
	e := startTLS(t, nil)
	c := e.client(t, e.ca.pool(), nil, nil)
	e.waitReady(t, c)

	resp, err := c.Get(e.base + "/healthz")
	if err != nil {
		t.Fatalf("TLS request with the right CA failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if resp.TLS.Version < tls.VersionTLS12 {
		t.Fatalf("negotiated TLS version %x, want >= 1.2", resp.TLS.Version)
	}
	if resp.TLS.Version != tls.VersionTLS13 {
		t.Logf("negotiated %x (TLS 1.3 expected with a modern Go client)", resp.TLS.Version)
	}

	// A client that does not trust the CA must refuse the server.
	if _, err := e.client(t, x509.NewCertPool(), nil, nil).Get(e.base + "/healthz"); err == nil {
		t.Fatal("client with an empty trust pool accepted the server certificate")
	}
}

func TestPlainHTTPIsRefusedOnTLSPort(t *testing.T) {
	e := startTLS(t, nil)
	e.waitReady(t, e.client(t, e.ca.pool(), nil, nil))

	plain := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{DisableKeepAlives: true}}
	resp, err := plain.Get(strings.Replace(e.base, "https://", "http://", 1) + "/healthz")
	if err != nil {
		return // connection refused/reset: refused
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == 200 || strings.Contains(string(body), `"status"`) {
		t.Fatalf("plain HTTP was served on the TLS port: %d %s", resp.StatusCode, body)
	}
}

func TestOldTLSVersionsRefused(t *testing.T) {
	e := startTLS(t, nil)
	e.waitReady(t, e.client(t, e.ca.pool(), nil, nil))

	for name, max := range map[string]uint16{"TLS1.1": tls.VersionTLS11, "TLS1.0": tls.VersionTLS10} {
		c := e.client(t, e.ca.pool(), nil, func(cfg *tls.Config) { cfg.MinVersion, cfg.MaxVersion = tls.VersionTLS10, max })
		_, err := c.Get(e.base + "/healthz")
		if err == nil {
			t.Fatalf("%s handshake succeeded; the floor must be TLS 1.2", name)
		}
		// The refusal must come from the server (alert), not from the client declining to try.
		if strings.Contains(err.Error(), "no supported versions") {
			t.Fatalf("%s: client never attempted the handshake, so the server floor is untested: %v", name, err)
		}
		t.Logf("%s refused as expected: %v", name, err)
	}
}

// ---- mutual TLS ----------------------------------------------------------------------------------------------

func withClientCA(c *Config, e *tlsEnv) { c.ClientCAFile = e.caPEM }

func TestMTLSRequiresValidClientCertificate(t *testing.T) {
	e := startTLS(t, withClientCA)
	good := e.client(t, e.ca.pool(), e.clientCert(t, e.ca, "node-1"), nil)
	e.waitReady(t, good) // also proves a certificate from the right CA is accepted

	resp, err := good.Get(e.base + "/v1/jobs")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 { // no token configured, so a verified certificate alone is enough
		t.Fatalf("client with a valid certificate got %d", resp.StatusCode)
	}

	// No client certificate: handshake must fail.
	if _, err := e.client(t, e.ca.pool(), nil, nil).Get(e.base + "/healthz"); err == nil {
		t.Fatal("client without a certificate was accepted under mTLS")
	}
	// Certificate from a different CA: rejected.
	other := newCA(t, "other-ca")
	if _, err := e.client(t, e.ca.pool(), e.clientCert(t, other, "intruder"), nil).Get(e.base + "/healthz"); err == nil {
		t.Fatal("client certificate from the wrong CA was accepted")
	}
}

func TestMTLSKeepsBearerTokenUnlessMTLSOnly(t *testing.T) {
	const tok = "s3cr3t-agent-token"
	e := startTLS(t, func(c *Config, e *tlsEnv) { withClientCA(c, e); c.Token = tok })
	c := e.client(t, e.ca.pool(), e.clientCert(t, e.ca, "node-1"), nil)
	e.waitReady(t, c)

	get := func(auth string) int {
		req, _ := http.NewRequest("GET", e.base+"/v1/jobs", nil)
		if auth != "" {
			req.Header.Set("Authorization", auth)
		}
		resp, err := c.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	if got := get(""); got != 401 {
		t.Errorf("valid cert but no bearer token: %d, want 401 (token is still required)", got)
	}
	if got := get("Bearer wrong"); got != 403 {
		t.Errorf("valid cert, wrong token: %d, want 403", got)
	}
	if got := get("Bearer " + tok); got != 200 {
		t.Errorf("valid cert and token: %d, want 200", got)
	}
}

func TestMTLSOnlyNeedsNoToken(t *testing.T) {
	e := startTLS(t, func(c *Config, e *tlsEnv) { withClientCA(c, e); c.Token = "ignored-token"; c.MTLSOnly = true })
	c := e.client(t, e.ca.pool(), e.clientCert(t, e.ca, "node-1"), nil)
	e.waitReady(t, c)

	resp, err := c.Get(e.base + "/v1/jobs")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("--mtls-only with a valid certificate got %d, want 200 without any bearer token", resp.StatusCode)
	}
	if _, err := e.client(t, e.ca.pool(), nil, nil).Get(e.base + "/v1/jobs"); err == nil {
		t.Fatal("--mtls-only accepted a client without a certificate")
	}
}

// ---- certificate hot reload ----------------------------------------------------------------------------------

func rotate(t *testing.T, e *tlsEnv, serial int64) {
	t.Helper()
	certPEM, keyPEM := e.ca.issue(t, "agent", serial, true)
	writeFile(t, e.cert, certPEM)
	writeFile(t, e.key, keyPEM)
}

func TestCertHotReloadOnDemand(t *testing.T) {
	e := startTLS(t, nil)
	c := e.client(t, e.ca.pool(), nil, nil)
	e.waitReady(t, c)
	if got := peerSerial(t, c, e.base); got != 100 {
		t.Fatalf("initial serial %d, want 100", got)
	}

	rotate(t, e, 200)
	if got := peerSerial(t, c, e.base); got != 100 {
		t.Fatalf("certificate changed before any reload (serial %d): files must not be re-read per request", got)
	}
	if err := e.s.ReloadCerts(); err != nil {
		t.Fatalf("reload failed: %v", err)
	}
	if got := peerSerial(t, c, e.base); got != 200 {
		t.Fatalf("serial after reload %d, want 200", got)
	}

	// A broken rotation (half-written file) must keep the previous certificate serving.
	writeFile(t, e.cert, []byte("not a certificate"))
	err := e.s.ReloadCerts()
	if err == nil {
		t.Fatal("reload of garbage reported success")
	}
	if strings.Contains(err.Error(), e.dir) {
		t.Fatalf("reload error leaks the cert path: %v", err)
	}
	if got := peerSerial(t, c, e.base); got != 200 {
		t.Fatalf("serial after failed reload %d, want the previous certificate (200)", got)
	}
}

// os errors for a missing file embed the full path ("open /dir/server.pem: ..."); neither the returned error nor
// the log may carry it.
func TestReloadMissingFileDoesNotLeakPath(t *testing.T) {
	e := startTLS(t, nil)
	c := e.client(t, e.ca.pool(), nil, nil)
	e.waitReady(t, c)

	if err := os.Remove(e.key); err != nil {
		t.Fatal(err)
	}
	err := e.s.ReloadCerts()
	if err == nil {
		t.Fatal("reload with a missing key file reported success")
	}
	if strings.Contains(err.Error(), e.dir) || strings.Contains(err.Error(), "server.key") {
		t.Fatalf("reload error leaks the key path: %v", err)
	}
	if out := e.logs.String(); strings.Contains(out, e.dir) || strings.Contains(out, "server.key") {
		t.Fatalf("log leaks the key path:\n%s", out)
	}
	if got := peerSerial(t, c, e.base); got != 100 {
		t.Fatalf("serial %d after a failed reload, want the previous certificate (100)", got)
	}
}

func TestCertHotReloadPeriodic(t *testing.T) {
	e := startTLS(t, func(c *Config, _ *tlsEnv) { c.CertReloadEvery = 50 * time.Millisecond })
	c := e.client(t, e.ca.pool(), nil, nil)
	e.waitReady(t, c)

	rotate(t, e, 300)
	deadline := time.Now().Add(5 * time.Second)
	for {
		if peerSerial(t, c, e.base) == 300 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("periodic reload never picked up the rotated certificate")
		}
		time.Sleep(25 * time.Millisecond)
	}
}

func TestReloadCertsWithoutTLSIsNoop(t *testing.T) {
	s, _ := newTestServer(t, nil)
	if err := s.ReloadCerts(); err != nil {
		t.Fatalf("ReloadCerts on a plain-HTTP agent: %v", err)
	}
}

// ---- logging -------------------------------------------------------------------------------------------------

func TestTLSLogsNeverContainTokenOrCertPaths(t *testing.T) {
	const tok = "tls-s3cr3t-token"
	e := startTLS(t, func(c *Config, e *tlsEnv) { withClientCA(c, e); c.Token = tok })
	c := e.client(t, e.ca.pool(), e.clientCert(t, e.ca, "node-1"), nil)
	e.waitReady(t, c)

	req, _ := http.NewRequest("POST", e.base+"/v1/jobs", strings.NewReader(validSpec))
	req.Header.Set("Authorization", "Bearer "+tok)
	if resp, err := c.Do(req); err == nil {
		resp.Body.Close()
	}
	// Provoke the failure paths too: a rejected client and a failed reload both log.
	_, _ = e.client(t, e.ca.pool(), nil, nil).Get(e.base + "/healthz")
	writeFile(t, e.cert, []byte("garbage"))
	_ = e.s.ReloadCerts()

	out := e.logs.String()
	if out == "" {
		t.Fatal("expected log output")
	}
	for _, leak := range []string{tok, e.dir, "server.pem", "server.key", "ca.pem", "org/model"} {
		if strings.Contains(out, leak) {
			t.Fatalf("log leaked %q:\n%s", leak, out)
		}
	}
}

// ---- configuration validation --------------------------------------------------------------------------------

func TestNewTLSConfigValidation(t *testing.T) {
	ca := newCA(t, "ca")
	dir := t.TempDir()
	cert, key, caFile := filepath.Join(dir, "c.pem"), filepath.Join(dir, "k.pem"), filepath.Join(dir, "ca.pem")
	cp, kp := ca.issue(t, "agent", 1, true)
	writeFile(t, cert, cp)
	writeFile(t, key, kp)
	writeFile(t, caFile, ca.pem)
	_, otherKey := ca.issue(t, "other", 2, true) // a key that does not match cert
	otherKeyFile := filepath.Join(dir, "other.key")
	writeFile(t, otherKeyFile, otherKey)
	garbageCA := filepath.Join(dir, "garbage-ca.pem")
	writeFile(t, garbageCA, []byte("nope"))
	quiet := log.New(io.Discard, "", 0)

	cases := []struct {
		name string
		cfg  Config
		ok   bool
	}{
		{"loopback plain", Config{Listen: "127.0.0.1:7070"}, true},
		{"loopback tls", Config{Listen: "127.0.0.1:7070", TLSCertFile: cert, TLSKeyFile: key}, true},
		{"non-loopback tls without credentials", Config{Listen: "0.0.0.0:7070", TLSCertFile: cert, TLSKeyFile: key}, false},
		{"non-loopback tls + token", Config{Listen: "0.0.0.0:7070", TLSCertFile: cert, TLSKeyFile: key, Token: "t"}, true},
		{"non-loopback mtls", Config{Listen: "0.0.0.0:7070", TLSCertFile: cert, TLSKeyFile: key, ClientCAFile: caFile}, true},
		{"non-loopback mtls-only", Config{Listen: "0.0.0.0:7070", TLSCertFile: cert, TLSKeyFile: key, ClientCAFile: caFile, MTLSOnly: true}, true},
		{"non-loopback token, no tls", Config{Listen: "0.0.0.0:7070", Token: "t"}, false},
		{"non-loopback insecure-http + token", Config{Listen: "0.0.0.0:7070", Token: "t", InsecureHTTP: true}, true},
		{"non-loopback insecure-http, no token", Config{Listen: "0.0.0.0:7070", InsecureHTTP: true}, false},
		{"cert without key", Config{Listen: "127.0.0.1:7070", TLSCertFile: cert}, false},
		{"key without cert", Config{Listen: "127.0.0.1:7070", TLSKeyFile: key}, false},
		{"mismatched key pair", Config{Listen: "127.0.0.1:7070", TLSCertFile: cert, TLSKeyFile: otherKeyFile}, false},
		{"missing cert file", Config{Listen: "127.0.0.1:7070", TLSCertFile: filepath.Join(dir, "nope.pem"), TLSKeyFile: key}, false},
		{"client CA without tls", Config{Listen: "127.0.0.1:7070", ClientCAFile: caFile}, false},
		{"mtls-only without client CA", Config{Listen: "127.0.0.1:7070", TLSCertFile: cert, TLSKeyFile: key, MTLSOnly: true}, false},
		{"garbage client CA", Config{Listen: "127.0.0.1:7070", TLSCertFile: cert, TLSKeyFile: key, ClientCAFile: garbageCA}, false},
		{"missing client CA", Config{Listen: "127.0.0.1:7070", TLSCertFile: cert, TLSKeyFile: key, ClientCAFile: filepath.Join(dir, "none.pem")}, false},
	}
	for _, c := range cases {
		c.cfg.Log = quiet
		_, err := New(c.cfg)
		if (err == nil) != c.ok {
			t.Errorf("%s: err=%v, want ok=%v", c.name, err, c.ok)
		}
		if err != nil && strings.Contains(err.Error(), dir) {
			t.Errorf("%s: error leaks a file path: %v", c.name, err)
		}
	}
}

func TestInsecureHTTPWarnsLoudly(t *testing.T) {
	var buf syncBuf
	if _, err := New(Config{Listen: "0.0.0.0:7070", Token: "t", InsecureHTTP: true, Log: log.New(&buf, "", 0)}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "WARNING") || !strings.Contains(buf.String(), "PLAIN HTTP") {
		t.Fatalf("expected a loud warning, got %q", buf.String())
	}
	buf = syncBuf{}
	if _, err := New(Config{Listen: "127.0.0.1:7070", Log: log.New(&buf, "", 0)}); err != nil {
		t.Fatal(err)
	}
	if buf.String() != "" {
		t.Fatalf("loopback plain HTTP should not warn, got %q", buf.String())
	}
}

// ---- error sanitizing ----------------------------------------------------------------------------------------

func TestSanitizeSpecError(t *testing.T) {
	cases := map[string]string{
		"windows url": "invalid spec: jsonschema validation failed with 'file:///C:/Users/me/OneDrive/Desktop/inference_ui/jobspec.json#'\n" +
			"- at '/spec/method': value must be one of 'lora', 'qlora', 'full'",
		"linux url": "invalid spec: jsonschema validation failed with 'file:///home/runner/work/tinyforge/jobspec.json#'\n" +
			"- at '/spec/data': additional properties 'bogus' not allowed",
		"stray drive path": `decode: open C:\Users\me\secret\job.yaml: boom`,
		"stray file url":   "oops file:///etc/passwd more",
	}
	for name, in := range cases {
		out := sanitizeSpecError(in)
		for _, bad := range []string{"file://", `C:\`, "C:/", "/home/", "/Users/", "OneDrive", "/etc/passwd"} {
			if strings.Contains(out, bad) {
				t.Errorf("%s: output still contains %q: %q", name, bad, out)
			}
		}
	}
	// Per-field messages and JSON pointers must survive.
	out := sanitizeSpecError(cases["windows url"])
	if !strings.Contains(out, "- at '/spec/method': value must be one of 'lora', 'qlora', 'full'") {
		t.Errorf("field message lost: %q", out)
	}
	if got := sanitizeSpecError("plain message"); got != "plain message" {
		t.Errorf("clean message was altered: %q", got)
	}
}

func TestInvalidSpecResponseHasNoLocalPaths(t *testing.T) {
	s, _ := newTestServer(t, nil)
	rec := do(s, "POST", "/v1/jobs", strings.Replace(validSpec, "method: lora", "method: nope", 1), nil)
	if rec.Code != 400 {
		t.Fatalf("status %d", rec.Code)
	}
	body := rec.Body.String()
	wd, _ := os.Getwd()
	for _, bad := range []string{"file://", filepath.ToSlash(wd), wd, "OneDrive"} {
		if bad != "" && strings.Contains(body, bad) {
			t.Fatalf("response leaks %q: %s", bad, body)
		}
	}
	if !strings.Contains(body, "/spec/method") {
		t.Fatalf("response lost the field message: %s", body)
	}
}
