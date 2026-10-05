package doctor

import (
	"fmt"
	"strings"
)

// Render produces the human-readable report.
func Render(r *Report) string {
	var b strings.Builder
	s := r.System
	fmt.Fprintf(&b, "forgectl doctor\n\nHost: %s/%s, %d CPUs", s.OS, s.Arch, s.CPUs)
	if s.RAMError == "" && s.RAMTotalBytes > 0 {
		fmt.Fprintf(&b, ", RAM %.1f GB (%.1f GB available)", float64(s.RAMTotalBytes)/gb, float64(s.RAMAvailBytes)/gb)
	}
	if s.DiskKnown {
		fmt.Fprintf(&b, ", %.1f GB free disk", float64(s.DiskFreeBytes)/gb)
	}
	b.WriteString("\n\nFindings:\n")
	for _, f := range r.Findings {
		fmt.Fprintf(&b, "  [%-5s] %s  %s\n", f.Level, f.Code, f.Message)
		if f.Fix != "" {
			fmt.Fprintf(&b, "                 fix: %s\n", f.Fix)
		}
	}
	fmt.Fprintf(&b, "\n%d error(s), %d warning(s)\n", r.Count(Error), r.Count(Warn))
	return b.String()
}
