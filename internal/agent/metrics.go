package agent

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"tinyforge.dev/forgectl/internal/doctor"
)

const miB = 1024 * 1024

// metrics serves Prometheus text exposition format 0.0.4, hand-written (no client library). Label values are
// bounded: GPU index, finding severity and job status only. GPU samples whose value the driver reported as
// N/A are omitted rather than emitted as zero.
func (s *Server) metrics(w http.ResponseWriter, _ *http.Request) {
	rep, at := s.snapshot()
	now := s.cfg.Now()
	var b strings.Builder

	family(&b, "forgectl_agent_up", "gauge", "1 while the agent process is serving.")
	sample(&b, "forgectl_agent_up", "", 1)
	family(&b, "forgectl_agent_info", "gauge", "Agent build information.")
	sample(&b, "forgectl_agent_info", `version="`+escapeLabel(s.cfg.Version)+`"`, 1)
	family(&b, "forgectl_agent_uptime_seconds", "gauge", "Seconds since the agent started.")
	sample(&b, "forgectl_agent_uptime_seconds", "", now.Sub(s.started).Seconds())
	counts := s.jobs.counts()
	family(&b, "forgectl_agent_jobs_queued", "gauge", "Jobs waiting to run (same value as forgectl_agent_jobs{status=\"queued\"}).")
	sample(&b, "forgectl_agent_jobs_queued", "", float64(counts[StatusQueued]))
	family(&b, "forgectl_agent_jobs", "gauge", "Jobs known to the agent by status.")
	for _, st := range allStatuses {
		sample(&b, "forgectl_agent_jobs", `status="`+st+`"`, float64(counts[st]))
	}
	s.durations.write(&b)

	if rep != nil {
		family(&b, "forgectl_doctor_report_age_seconds", "gauge", "Age of the cached doctor report.")
		sample(&b, "forgectl_doctor_report_age_seconds", "", now.Sub(at).Seconds())
		family(&b, "forgectl_doctor_findings", "gauge", "Findings in the cached doctor report by severity.")
		for _, lvl := range []doctor.Level{doctor.Info, doctor.Warn, doctor.Error} {
			sample(&b, "forgectl_doctor_findings", `severity="`+string(lvl)+`"`, float64(rep.Count(lvl)))
		}
		family(&b, "forgectl_node_gpu_count", "gauge", "NVIDIA GPUs visible to nvidia-smi.")
		sample(&b, "forgectl_node_gpu_count", "", float64(len(rep.GPUs)))

		family(&b, "forgectl_node_gpu_memory_total_bytes", "gauge", "GPU memory capacity.")
		for _, g := range rep.GPUs {
			if g.MemTotalMiB != nil {
				sample(&b, "forgectl_node_gpu_memory_total_bytes", gpuLabel(g), *g.MemTotalMiB*miB)
			}
		}
		family(&b, "forgectl_node_gpu_memory_used_bytes", "gauge", "GPU memory in use.")
		for _, g := range rep.GPUs {
			if g.MemUsedMiB != nil {
				sample(&b, "forgectl_node_gpu_memory_used_bytes", gpuLabel(g), *g.MemUsedMiB*miB)
			}
		}
		family(&b, "forgectl_node_gpu_temperature_celsius", "gauge", "GPU core temperature.")
		for _, g := range rep.GPUs {
			if g.TempC != nil {
				sample(&b, "forgectl_node_gpu_temperature_celsius", gpuLabel(g), *g.TempC)
			}
		}
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_, _ = w.Write([]byte(b.String()))
}

func gpuLabel(g doctor.GPU) string { return `gpu="` + strconv.Itoa(g.Index) + `"` }

func family(b *strings.Builder, name, typ, help string) {
	fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, typ)
}

func sample(b *strings.Builder, name, labels string, v float64) {
	if labels != "" {
		labels = "{" + labels + "}"
	}
	fmt.Fprintf(b, "%s%s %s\n", name, labels, strconv.FormatFloat(v, 'g', -1, 64))
}

// escapeLabel escapes backslash, double quote and newline per the exposition format.
func escapeLabel(v string) string {
	return strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`).Replace(v)
}
