package netcheck

import "fmt"

// localEndpoint is one of this project's planned loopback endpoints (see docs/networking.md section 2).
type localEndpoint struct {
	id   string
	name string
	port int
}

var localEndpoints = []localEndpoint{
	{"local-api", "tinyforge API", 8000},
	{"local-agent", "forgectl agent", 7070},
	{"local-prometheus", "Prometheus", 9090},
	{"local-grafana", "Grafana", 3000},
	{"local-gpu-exporter", "NVIDIA GPU exporter", 9835},
}

// LocalPreset checks this machine's own endpoints on loopback. A service that is not running is reported as INFO
// ("not listening"), never as a failure, because any of them may legitimately be down.
func LocalPreset() []Check {
	checks := make([]Check, 0, len(localEndpoints))
	for _, e := range localEndpoints {
		c := Check{
			ID: e.id, Kind: KindTCP, Target: fmt.Sprintf("127.0.0.1:%d", e.port),
			Expect: Reachable, Severity: SevInfo,
		}
		if err := c.resolve(""); err != nil { // cannot happen: the targets above are constants
			panic("netcheck: bad local preset: " + err.Error())
		}
		c.hint = "not listening (" + e.name + ")"
		checks = append(checks, c)
	}
	return checks
}
