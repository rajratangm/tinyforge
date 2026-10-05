package netcheck

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestExampleFileLoads(t *testing.T) {
	checks, err := Load(filepath.Join("testdata", "example.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	kinds := map[string]int{}
	for _, c := range checks {
		kinds[c.Kind]++
	}
	want := map[string]int{KindDNS: 1, KindTCP: 1, KindTLS: 1, KindHTTP: 1, KindEgressDeny: 2}
	for k, n := range want {
		if kinds[k] != n {
			t.Errorf("kind %s: got %d checks, want %d (%v)", k, kinds[k], n, kinds)
		}
	}
}

func TestDefaultsAreResolved(t *testing.T) {
	cs := mustParse(t, `
checks:
  - {id: a, kind: tcp, target: "h.example:80"}
  - {id: b, kind: egress-deny, target: "1.2.3.4:443"}
  - {id: c, kind: http, target: "https://h.example/x"}
  - {id: d, kind: tls, target: "h.example:443"}
`)
	if cs[0].Expect != Reachable || cs[0].Severity != SevError {
		t.Errorf("tcp defaults: %+v", cs[0])
	}
	if cs[1].Expect != Blocked {
		t.Errorf("egress-deny must default to blocked, got %q", cs[1].Expect)
	}
	if cs[2].port != 443 || cs[2].ExpectStatus != 200 || cs[2].host != "h.example" {
		t.Errorf("http defaults: port=%d status=%d host=%s", cs[2].port, cs[2].ExpectStatus, cs[2].host)
	}
	if cs[3].warnDays != defaultWarnDay || cs[3].minTLS == 0 {
		t.Errorf("tls defaults: warn=%d min=%d", cs[3].warnDays, cs[3].minTLS)
	}
}

func TestParseRejections(t *testing.T) {
	many := "checks:\n"
	for i := 0; i <= MaxChecks; i++ {
		many += fmt.Sprintf("  - {id: c%d, kind: tcp, target: \"h.example:80\"}\n", i)
	}
	cases := []struct{ name, doc, want string }{
		{"empty", "", "empty"},
		{"not yaml", "::: not yaml :::", "parse"},
		{"no checks", "checks: []", "at least one"},
		{"unknown top-level field", "bogus: 1\nchecks:\n  - {id: a, kind: tcp, target: \"h:1\"}", "bogus"},
		{"unknown check field", "checks:\n  - {id: a, kind: tcp, target: \"h:1\", bogus: 1}", "bogus"},
		{"two documents", "checks:\n  - {id: a, kind: tcp, target: \"h:1\"}\n---\nchecks: []", "one YAML document"},
		{"wrong apiVersion", "apiVersion: v9\nchecks:\n  - {id: a, kind: tcp, target: \"h:1\"}", "apiVersion"},
		{"wrong kind", "kind: Other\nchecks:\n  - {id: a, kind: tcp, target: \"h:1\"}", "kind"},
		{"missing id", "checks:\n  - {kind: tcp, target: \"h:1\"}", "id"},
		{"bad id", "checks:\n  - {id: Bad_ID, kind: tcp, target: \"h:1\"}", "id"},
		{"duplicate id", "checks:\n  - {id: a, kind: tcp, target: \"h:1\"}\n  - {id: a, kind: tcp, target: \"h:2\"}", "duplicate"},
		{"bad kind", "checks:\n  - {id: a, kind: icmp, target: \"h\"}", "kind"},
		{"CIDR target", "checks:\n  - {id: a, kind: tcp, target: \"10.0.0.0/24:443\"}", "CIDR"},
		{"CIDR without port", "checks:\n  - {id: a, kind: tcp, target: \"10.0.0.0/24\"}", "target"},
		{"port range", "checks:\n  - {id: a, kind: tcp, target: \"h.example:80-90\"}", "port"},
		{"port list", "checks:\n  - {id: a, kind: tcp, target: \"h.example:80,443\"}", "target"},
		{"wildcard host", "checks:\n  - {id: a, kind: tcp, target: \"*.example.com:443\"}", "wildcard"},
		{"host list", "checks:\n  - {id: a, kind: tcp, target: \"a.example b.example:443\"}", "target"},
		{"port zero", "checks:\n  - {id: a, kind: tcp, target: \"h.example:0\"}", "port"},
		{"port too big", "checks:\n  - {id: a, kind: tcp, target: \"h.example:70000\"}", "port"},
		{"no port", "checks:\n  - {id: a, kind: tcp, target: \"h.example\"}", "host:port"},
		{"url credentials", "checks:\n  - {id: a, kind: http, target: \"https://user:pw@h.example/\"}", "credentials"},
		{"url scheme", "checks:\n  - {id: a, kind: http, target: \"ftp://h.example/\"}", "scheme"},
		{"url no host", "checks:\n  - {id: a, kind: http, target: \"https:///x\"}", "URL"},
		{"dns IP literal", "checks:\n  - {id: a, kind: dns, target: \"10.0.0.1\"}", "hostname"},
		{"dns CIDR", "checks:\n  - {id: a, kind: dns, target: \"10.0.0.0/8\"}", "hostname"},
		{"expectCIDR on tcp", "checks:\n  - {id: a, kind: tcp, target: \"h:1\", expectCIDR: 10.0.0.0/8}", "does not apply"},
		{"caFile on dns", "checks:\n  - {id: a, kind: dns, target: h.example, caFile: x.pem}", "does not apply"},
		{"expectStatus on tls", "checks:\n  - {id: a, kind: tls, target: \"h:1\", expectStatus: 200}", "does not apply"},
		{"bad expectCIDR", "checks:\n  - {id: a, kind: dns, target: h.example, expectCIDR: nope}", "CIDR"},
		{"bad timeout", "checks:\n  - {id: a, kind: tcp, target: \"h:1\", timeout: soon}", "timeout"},
		{"timeout too long", "checks:\n  - {id: a, kind: tcp, target: \"h:1\", timeout: 1h}", "timeout"},
		{"TLS 1.1", "checks:\n  - {id: a, kind: tls, target: \"h:1\", minTLSVersion: \"1.1\"}", "minTLSVersion"},
		{"bad expect", "checks:\n  - {id: a, kind: tcp, target: \"h:1\", expect: maybe}", "expect"},
		{"bad severity", "checks:\n  - {id: a, kind: tcp, target: \"h:1\", severity: fatal}", "severity"},
		{"egress-deny expect reachable", "checks:\n  - {id: a, kind: egress-deny, target: \"h:1\", expect: reachable}", "egress-deny"},
		{"status out of range", "checks:\n  - {id: a, kind: http, target: \"http://h.example/\", expectStatus: 99}", "expectStatus"},
		{"missing caFile", "checks:\n  - {id: a, kind: tls, target: \"h:1\", caFile: /no/such/ca.pem}", "caFile"},
		{"too many checks", many, "limit"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := Parse([]byte(c.doc), t.TempDir())
			if err == nil {
				t.Fatal("expected an error, got none")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not mention %q", err, c.want)
			}
		})
	}
}

func TestParseReportsEveryBadCheck(t *testing.T) {
	_, err := Parse([]byte("checks:\n  - {id: a, kind: icmp, target: x}\n  - {id: b, kind: tcp, target: \"10.0.0.0/8:1\"}"), "")
	if err == nil || !strings.Contains(err.Error(), `"a"`) || !strings.Contains(err.Error(), `"b"`) {
		t.Fatalf("both failing checks should be reported, got: %v", err)
	}
}

func TestJSONInputAccepted(t *testing.T) {
	cs := mustParse(t, `{"checks":[{"id":"j","kind":"tcp","target":"h.example:80"}]}`)
	if len(cs) != 1 || cs[0].ID != "j" {
		t.Fatalf("JSON checks file not parsed: %+v", cs)
	}
}

func TestCAFileResolvesRelativeToTheChecksFile(t *testing.T) {
	dir := t.TempDir()
	ca := newCA(t, nowT().Add(-time.Hour), nowT().Add(time.Hour))
	ca.writeCA(t, dir, "ca.pem")
	path := filepath.Join(dir, "checks.yaml")
	body := "checks:\n  - {id: a, kind: tls, target: \"h.example:443\", caFile: ca.pem}\n"
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	cs, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cs[0].pool == nil {
		t.Error("caFile was not loaded into a pool")
	}
	// a file that is not PEM is rejected up front
	if err := os.WriteFile(filepath.Join(dir, "ca.pem"), []byte("not a cert"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil || !strings.Contains(err.Error(), "PEM") {
		t.Errorf("non-PEM caFile should be rejected, got %v", err)
	}
}

func TestLoadMissingFile(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "nope.yaml")); err == nil {
		t.Fatal("expected an error for a missing file")
	}
}
