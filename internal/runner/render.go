package runner

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// RenderEvent turns one worker stdout line into a compact human line. ok is false for anything that is not a
// JSON object with a known "event" (garbage, unknown events, events from a newer worker): consumers must ignore those.
func RenderEvent(line []byte, elapsed time.Duration) (text string, ok bool) {
	var m map[string]any
	if err := json.Unmarshal(line, &m); err != nil {
		return "", false
	}
	ev, _ := m["event"].(string)
	var body string
	switch ev {
	case "started":
		body = "started" + fields(m, "params", "total_params", "device", "precision", "quant", "max_steps")
	case "resumed":
		body = "resumed from step " + val(m, "step")
	case "step":
		body = "step " + val(m, "step") + fields(m, "loss", "lr", "grad_norm", "tok_per_s", "peak_mem_gb")
	case "eval":
		body = "eval step " + val(m, "step") + fields(m, "val_loss", "train_loss", "val_ppl")
	case "diagnostic":
		body = fmt.Sprintf("[%s] %s %s", orDash(str(m, "level")), orDash(str(m, "code")), str(m, "message"))
		if fix := str(m, "fix"); fix != "" {
			body += "  fix: " + fix
		}
	case "gate":
		body = fmt.Sprintf("gate %s %s %s: %s", str(m, "metric"), str(m, "op"), val(m, "value"), gateVerdict(m))
	case "finished":
		body = "finished" + fields(m, "steps", "best_val_loss", "best_val_ppl", "trainable_params", "peak_mem_gb")
	case "failed":
		body = "FAILED: " + orDash(str(m, "reason"))
	default:
		return "", false
	}
	return fmt.Sprintf("[%s] %s", clock(elapsed), body), true
}

func gateVerdict(m map[string]any) string {
	if passed, _ := m["passed"].(bool); passed {
		return "PASS (observed " + val(m, "observed") + ")"
	}
	if m["observed"] == nil {
		return "FAIL (could not be evaluated)"
	}
	return "FAIL (observed " + val(m, "observed") + ")"
}

// fields renders "k=v" pairs for the keys that are present, in the given order.
func fields(m map[string]any, keys ...string) string {
	var b strings.Builder
	for _, k := range keys {
		if v, present := m[k]; present && v != nil {
			b.WriteString(" " + k + "=" + format(v))
		}
	}
	return b.String()
}

func val(m map[string]any, k string) string {
	v, present := m[k]
	if !present || v == nil {
		return "?"
	}
	return format(v)
}

func format(v any) string {
	switch t := v.(type) {
	case float64:
		return strconv.FormatFloat(t, 'g', 4, 64)
	case string:
		return t
	case bool:
		return strconv.FormatBool(t)
	default:
		b, _ := json.Marshal(t)
		return string(b)
	}
}

func str(m map[string]any, k string) string {
	s, _ := m[k].(string)
	return s
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func clock(d time.Duration) string {
	s := int(d.Seconds())
	if s < 0 {
		s = 0
	}
	return fmt.Sprintf("%02d:%02d", s/60, s%60)
}
