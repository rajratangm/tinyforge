package doctor

import (
	"context"
	"fmt"
	"strconv"
	"strings"
)

// GPU is one device from nvidia-smi. Pointer fields are nil when the driver reports [N/A]/[Not Supported].
type GPU struct {
	Index        int      `json:"index"`
	Name         string   `json:"name"`
	MemTotalMiB  *float64 `json:"mem_total_mib,omitempty"`
	MemUsedMiB   *float64 `json:"mem_used_mib,omitempty"`
	Driver       string   `json:"driver,omitempty"`
	TempC        *float64 `json:"temp_c,omitempty"`
	PowerW       *float64 `json:"power_w,omitempty"`
	PowerLimitW  *float64 `json:"power_limit_w,omitempty"`
	ThrottleMask *uint64  `json:"throttle_mask,omitempty"`
	PCIeGenCur   *int     `json:"pcie_gen_current,omitempty"`
	PCIeGenMax   *int     `json:"pcie_gen_max,omitempty"`
	PCIeWidthCur *int     `json:"pcie_width_current,omitempty"`
	PCIeWidthMax *int     `json:"pcie_width_max,omitempty"`
}

// Throttle bits from nvidia-smi clocks_throttle_reasons (NVML clocksEventReasons).
const (
	ThrottleSWPowerCap   uint64 = 0x04
	ThrottleHWSlowdown   uint64 = 0x08
	ThrottleSWThermal    uint64 = 0x20
	ThrottleHWThermal    uint64 = 0x40
	ThrottleHWPowerBrake uint64 = 0x80
)

var throttleNames = []struct {
	bit  uint64
	name string
}{
	{0x01, "GPU idle"}, {0x02, "application clocks setting"}, {ThrottleSWPowerCap, "software power cap"},
	{ThrottleHWSlowdown, "hardware slowdown"}, {0x10, "sync boost"}, {ThrottleSWThermal, "software thermal slowdown"},
	{ThrottleHWThermal, "hardware thermal slowdown"}, {ThrottleHWPowerBrake, "hardware power brake"},
	{0x100, "display clocks setting"},
}

func ThrottleReasons(mask uint64) []string {
	var out []string
	for _, t := range throttleNames {
		if mask&t.bit != 0 {
			out = append(out, t.name)
		}
	}
	return out
}

const queryFields = "index,name,memory.total,memory.used,driver_version,temperature.gpu,power.draw,power.limit," +
	"%s,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"

// Older drivers call the throttle field clocks_throttle_reasons.active, newer ones clocks_event_reasons.active.
var throttleFields = []string{"clocks_throttle_reasons.active", "clocks_event_reasons.active"}

func queryArgs(throttleField string) []string {
	return []string{"--query-gpu=" + fmt.Sprintf(queryFields, throttleField), "--format=csv,noheader,nounits"}
}

// queryGPUs runs nvidia-smi, retrying with the alternate throttle field name if the driver rejects the first.
func queryGPUs(ctx context.Context, rn Runner, smi string) ([]GPU, string, error) {
	var out string
	var err error
	for _, f := range throttleFields {
		out, err = rn.Run(ctx, smi, queryArgs(f)...)
		if err == nil {
			g, perr := ParseGPUs(out)
			return g, out, perr
		}
		if !strings.Contains(strings.ToLower(out), "not a valid field") {
			break
		}
	}
	return nil, out, err
}

const csvColumns = 13

// ParseGPUs parses `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` output.
func ParseGPUs(csv string) ([]GPU, error) {
	var gpus []GPU
	for n, line := range strings.Split(csv, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		f := strings.Split(line, ",")
		if len(f) != csvColumns {
			return nil, fmt.Errorf("line %d: expected %d columns, got %d", n+1, csvColumns, len(f))
		}
		for i := range f {
			f[i] = strings.TrimSpace(f[i])
		}
		idx, err := strconv.Atoi(f[0])
		if err != nil {
			return nil, fmt.Errorf("line %d: bad GPU index %q", n+1, f[0])
		}
		g := GPU{
			Index: idx, Name: f[1], Driver: unavailable(f[4]),
			MemTotalMiB: optFloat(f[2]), MemUsedMiB: optFloat(f[3]), TempC: optFloat(f[5]),
			PowerW: optFloat(f[6]), PowerLimitW: optFloat(f[7]), ThrottleMask: optHex(f[8]),
			PCIeGenCur: optInt(f[9]), PCIeGenMax: optInt(f[10]), PCIeWidthCur: optInt(f[11]), PCIeWidthMax: optInt(f[12]),
		}
		gpus = append(gpus, g)
	}
	return gpus, nil
}

// unavailable returns "" for [N/A], [Not Supported], [Insufficient Permissions], N/A, etc.
func unavailable(s string) string {
	if s == "" || strings.EqualFold(s, "n/a") || (strings.HasPrefix(s, "[") && strings.HasSuffix(s, "]")) {
		return ""
	}
	return s
}

func optFloat(s string) *float64 {
	if s = unavailable(s); s == "" {
		return nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return nil
	}
	return &v
}

func optInt(s string) *int {
	if s = unavailable(s); s == "" {
		return nil
	}
	v, err := strconv.Atoi(s)
	if err != nil {
		return nil
	}
	return &v
}

func optHex(s string) *uint64 {
	if s = unavailable(s); s == "" {
		return nil
	}
	v, err := strconv.ParseUint(strings.TrimPrefix(strings.ToLower(s), "0x"), 16, 64)
	if err != nil {
		return nil
	}
	return &v
}

// Summary is the one-line description used in the report.
func (g GPU) Summary() string {
	var b strings.Builder
	fmt.Fprintf(&b, "GPU %d: %s", g.Index, g.Name)
	if g.MemTotalMiB != nil {
		fmt.Fprintf(&b, ", %.0f MiB VRAM", *g.MemTotalMiB)
		if g.MemUsedMiB != nil {
			fmt.Fprintf(&b, " (%.0f used)", *g.MemUsedMiB)
		}
	}
	if g.Driver != "" {
		fmt.Fprintf(&b, ", driver %s", g.Driver)
	}
	if g.TempC != nil {
		fmt.Fprintf(&b, ", %.0f C", *g.TempC)
	}
	if g.PowerW != nil {
		fmt.Fprintf(&b, ", %.1f W", *g.PowerW)
		if g.PowerLimitW != nil {
			fmt.Fprintf(&b, " / %.0f W limit", *g.PowerLimitW)
		}
	}
	if g.PCIeGenCur != nil && g.PCIeWidthCur != nil {
		fmt.Fprintf(&b, ", PCIe gen%d x%d", *g.PCIeGenCur, *g.PCIeWidthCur)
		if g.PCIeGenMax != nil && g.PCIeWidthMax != nil {
			fmt.Fprintf(&b, " (max gen%d x%d)", *g.PCIeGenMax, *g.PCIeWidthMax)
		}
	}
	return b.String()
}
