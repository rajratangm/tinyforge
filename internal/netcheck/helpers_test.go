package netcheck

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// pki is a throwaway ECDSA P-256 certificate authority.
type pki struct {
	cert   *x509.Certificate
	key    *ecdsa.PrivateKey
	pem    []byte
	serial atomic.Int64
}

func newCA(t *testing.T, notBefore, notAfter time.Time) *pki {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "netcheck test CA"},
		NotBefore: notBefore, NotAfter: notAfter, IsCA: true, BasicConstraintsValid: true,
		KeyUsage: x509.KeyUsageCertSign | x509.KeyUsageDigitalSignature,
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

// leaf issues a server certificate for the given names and addresses.
func (p *pki) leaf(t *testing.T, dns []string, ips []net.IP, notBefore, notAfter time.Time) tls.Certificate {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(p.serial.Add(1) + 100), Subject: pkix.Name{CommonName: "netcheck test leaf"},
		NotBefore: notBefore, NotAfter: notAfter, DNSNames: dns, IPAddresses: ips,
		KeyUsage: x509.KeyUsageDigitalSignature, ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, p.cert, &key.PublicKey, p.key)
	if err != nil {
		t.Fatal(err)
	}
	return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key}
}

// writeCA stores the CA bundle on disk and returns its path (for caFile).
func (p *pki) writeCA(t *testing.T, dir, name string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, p.pem, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

var loopback = []net.IP{net.ParseIP("127.0.0.1")}

// startTLS runs a bare TLS server that completes the handshake and closes. stall makes it accept and never respond.
func startTLS(t *testing.T, cert tls.Certificate, maxVersion uint16, stall bool) (host string, port int) {
	t.Helper()
	cfg := &tls.Config{Certificates: []tls.Certificate{cert}, MaxVersion: maxVersion}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	var conns sync.WaitGroup
	t.Cleanup(func() {
		close(done)
		ln.Close()
		conns.Wait()
	})
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			conns.Add(1)
			go func() {
				defer conns.Done()
				defer c.Close()
				if stall {
					<-done
					return
				}
				_ = c.SetDeadline(time.Now().Add(5 * time.Second))
				_ = tls.Server(c, cfg).Handshake()
			}()
		}
	}()
	a := ln.Addr().(*net.TCPAddr)
	return "127.0.0.1", a.Port
}

// startHTTPS runs an HTTPS server with the given certificate.
func startHTTPS(t *testing.T, cert tls.Certificate, h http.Handler) *httptest.Server {
	t.Helper()
	s := httptest.NewUnstartedServer(h)
	s.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
	s.StartTLS()
	t.Cleanup(s.Close)
	return s
}

func hostPort(t *testing.T, addr string) (string, int) {
	t.Helper()
	h, p, err := net.SplitHostPort(addr)
	if err != nil {
		t.Fatal(err)
	}
	n, _ := net.LookupPort("tcp", p)
	return h, n
}

// closedPort returns a loopback port with nothing listening on it.
func closedPort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()
	return port
}

// openPort returns a loopback listener that accepts and drops connections.
func openPort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}()
	return ln.Addr().(*net.TCPAddr).Port
}

// fakeResolver answers from a map; a missing name is NXDOMAIN.
type fakeResolver map[string][]string

func (f fakeResolver) LookupHost(_ context.Context, host string) ([]string, error) {
	if a, ok := f[host]; ok {
		return append([]string(nil), a...), nil
	}
	return nil, &net.DNSError{Err: "no such host", Name: host, IsNotFound: true}
}

// blockingDialer never connects: it waits for the context to end, like a firewall dropping packets.
func blockingDialer(ctx context.Context, _, _ string) (net.Conn, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

// mustParse builds validated checks from YAML.
func mustParse(t *testing.T, doc string) []Check {
	t.Helper()
	checks, err := Parse([]byte(doc), t.TempDir())
	if err != nil {
		t.Fatalf("parse: %v\n%s", err, doc)
	}
	return checks
}

// one returns the single result of a one-check run.
func one(t *testing.T, doc string, o Options) Result {
	t.Helper()
	rep := Run(context.Background(), mustParse(t, doc), o)
	if len(rep.Results) != 1 {
		t.Fatalf("want 1 result, got %d", len(rep.Results))
	}
	return rep.Results[0]
}
