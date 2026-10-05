package netcheck

import (
	"encoding/json"
	"fmt"
	"strings"
	"text/tabwriter"
)

// JSON renders the report as indented JSON. Results are already sorted by id, so the output is stable
// apart from the per-check durations.
func (r Report) JSON() ([]byte, error) { return json.MarshalIndent(r, "", "  ") }

// Render formats the report as a table followed by a summary line.
func Render(r Report) string {
	var b strings.Builder
	tw := tabwriter.NewWriter(&b, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "STATUS\tID\tKIND\tTARGET\tEXPECT\tOBSERVED\tDETAIL")
	for _, x := range r.Results {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			strings.ToUpper(x.Status), x.ID, x.Kind, x.Target, x.Expect, x.Observed, oneLine(x.Detail))
	}
	_ = tw.Flush()
	fmt.Fprintf(&b, "\n%d pass, %d warn, %d info, %d fail\n", r.Summary.Pass, r.Summary.Warn, r.Summary.Info, r.Summary.Fail)
	return b.String()
}

func oneLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
}
