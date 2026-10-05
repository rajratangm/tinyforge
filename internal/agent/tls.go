package agent

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io/fs"
	"net"
	"os"
	"regexp"
	"sync"
)

// Security describes how a listener is protected; CheckListen decides whether that is acceptable for its address.
type Security struct {
	TLS          bool // server certificate configured
	MTLS         bool // client certificates required (implies TLS)
	Token        bool // a bearer token is enforced on the API
	InsecureHTTP bool // operator explicitly accepted plain HTTP on a non-loopback address
}

// CheckListen enforces the listener policy. Loopback addresses are always allowed. A non-loopback address (an
// empty host such as ":7070" means all interfaces) needs TLS plus credentials (a bearer token or client
// certificates). The only escape hatch is InsecureHTTP, which waives TLS but still requires a bearer token.
func CheckListen(addr string, sec Security) error {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("invalid listen address %q: %w", addr, err)
	}
	if isLoopbackHost(host) {
		return nil
	}
	if sec.TLS {
		if sec.Token || sec.MTLS {
			return nil
		}
		return fmt.Errorf("refusing to listen on non-loopback address %q without credentials: "+
			"set --token-file or FORGECTL_AGENT_TOKEN, or require client certificates with --tls-client-ca", addr)
	}
	if sec.InsecureHTTP {
		if sec.Token {
			return nil
		}
		return fmt.Errorf("refusing to listen on non-loopback address %q without a bearer token, even with --insecure-http", addr)
	}
	return fmt.Errorf("refusing to listen on non-loopback address %q without TLS: set --tls-cert and --tls-key "+
		"(plus a token or --tls-client-ca), listen on 127.0.0.1, or pass --insecure-http with a token if you accept "+
		"unencrypted traffic", addr)
}

func isLoopbackHost(h string) bool {
	if h == "localhost" {
		return true
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}

// certReloader serves the current key pair and swaps it atomically on Reload. A failed reload keeps the old
// pair, so a half-written rotation never takes the agent down.
type certReloader struct {
	certFile, keyFile string

	mu   sync.RWMutex
	cert *tls.Certificate
}

func newCertReloader(certFile, keyFile string) (*certReloader, error) {
	r := &certReloader{certFile: certFile, keyFile: keyFile}
	if err := r.Reload(); err != nil {
		return nil, err
	}
	return r, nil
}

// Reload re-reads the certificate and key from disk. The returned error never contains file paths.
func (r *certReloader) Reload() error {
	c, err := tls.LoadX509KeyPair(r.certFile, r.keyFile)
	if err != nil {
		return sanitizeFileError(err)
	}
	r.mu.Lock()
	r.cert = &c
	r.mu.Unlock()
	return nil
}

func (r *certReloader) getCertificate(*tls.ClientHelloInfo) (*tls.Certificate, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.cert, nil
}

// sanitizeFileError keeps the failure class (missing, unreadable, invalid) but drops paths from os errors.
func sanitizeFileError(err error) error {
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return errors.New("certificate or key file not found")
	case errors.Is(err, fs.ErrPermission):
		return errors.New("certificate or key file is not readable")
	default:
		var pe *fs.PathError
		if errors.As(err, &pe) {
			return errors.New("certificate or key file cannot be read")
		}
		return errors.New("certificate and key do not form a valid pair")
	}
}

// buildTLSConfig returns the server TLS config, or nil when no certificate is configured. TLS 1.2 is the floor
// and Go's default cipher suites are used (TLS 1.3 is preferred automatically); nothing custom or weaker.
func buildTLSConfig(cfg Config) (*tls.Config, *certReloader, error) {
	if cfg.TLSCertFile == "" && cfg.TLSKeyFile == "" {
		if cfg.ClientCAFile != "" || cfg.MTLSOnly {
			return nil, nil, errors.New("client certificate options need --tls-cert and --tls-key")
		}
		return nil, nil, nil
	}
	if cfg.TLSCertFile == "" || cfg.TLSKeyFile == "" {
		return nil, nil, errors.New("--tls-cert and --tls-key must be set together")
	}
	if cfg.MTLSOnly && cfg.ClientCAFile == "" {
		return nil, nil, errors.New("--mtls-only needs --tls-client-ca")
	}
	rl, err := newCertReloader(cfg.TLSCertFile, cfg.TLSKeyFile)
	if err != nil {
		return nil, nil, fmt.Errorf("tls: %w", err)
	}
	tc := &tls.Config{MinVersion: tls.VersionTLS12, GetCertificate: rl.getCertificate}
	if cfg.ClientCAFile != "" {
		pem, err := os.ReadFile(cfg.ClientCAFile)
		if err != nil {
			return nil, nil, fmt.Errorf("tls client CA: %w", sanitizeFileError(err))
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, nil, errors.New("tls client CA: no valid PEM certificates found")
		}
		tc.ClientCAs = pool
		tc.ClientAuth = tls.RequireAndVerifyClientCert
	}
	return tc, rl, nil
}

// ReloadCerts re-reads the server certificate and key (call it on SIGHUP). It is a no-op without TLS.
func (s *Server) ReloadCerts() error {
	if s.certs == nil {
		return nil
	}
	if err := s.certs.Reload(); err != nil {
		s.cfg.Log.Printf("tls: certificate reload failed, keeping the previous certificate: %v", err)
		return err
	}
	s.cfg.Log.Printf("tls: certificate reloaded")
	return nil
}

var (
	// jobspec reports validation failures as "jsonschema validation failed with '<file:///local/path>#'" followed
	// by one "- at '/pointer': message" line per problem. The prefix names a local path, so drop it.
	schemaClause = regexp.MustCompile(`jsonschema validation failed with '[^']*'`)
	fileURL      = regexp.MustCompile(`file:///[^\s'"]*`)
	winPath      = regexp.MustCompile(`[A-Za-z]:[\\/][^\s'"]*`)
)

// sanitizeSpecError removes local file paths from a spec validation error while keeping the per-field messages
// (JSON pointers such as /spec/method are not paths and are left alone).
func sanitizeSpecError(msg string) string {
	msg = schemaClause.ReplaceAllString(msg, "validation failed")
	msg = fileURL.ReplaceAllString(msg, "<path>")
	return winPath.ReplaceAllString(msg, "<path>")
}
