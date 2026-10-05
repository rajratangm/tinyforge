package runner

import "fmt"

// Exit codes from spec/worker-contract.md.
const (
	ExitOK        = 0
	ExitCrash     = 1
	ExitSpec      = 2
	ExitGate      = 3
	ExitFit       = 4
	ExitDiverged  = 5
	ExitPreempted = 75
)

// Describe is the one-line human summary for a worker exit code.
func Describe(code int, dryRun bool) string {
	switch code {
	case ExitOK:
		if dryRun {
			return "dry run OK: the spec is valid and maps to a worker config; nothing was trained"
		}
		return "succeeded: training finished and all gates passed"
	case ExitCrash:
		return "worker crashed (exit 1): see stderr above"
	case ExitSpec:
		return "spec invalid or unsupported by this worker (exit 2): not retried"
	case ExitGate:
		return "gate failed (exit 3): artifacts were kept, nothing is promoted"
	case ExitFit:
		return "out of memory or the plan does not fit (exit 4): use a larger GPU or lower maxLen/batchSize"
	case ExitDiverged:
		return "training diverged (exit 5): lower the learning rate or check the data"
	case ExitPreempted:
		return "preempted (exit 75): the last checkpoint is kept; run the same command again to resume"
	default:
		return fmt.Sprintf("worker exited with code %d", code)
	}
}
