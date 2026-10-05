//go:build windows

package runner

import (
	"os"
	"syscall"
)

func sysProcAttr() *syscall.SysProcAttr { return nil }

// Windows has no SIGTERM for child processes. LIMITATION: a console Ctrl-C is delivered by Windows to every process
// on the console, so the worker already received it and forgectl must not kill it (nil here). Any other signal
// (for example one sent from another process) can only be forwarded as a hard Kill, which gives the worker no chance
// to checkpoint; the run then exits non-zero and resumes from the last periodic checkpoint.
func forward(p *os.Process, sig os.Signal) error {
	if sig == os.Interrupt {
		return nil
	}
	return p.Kill()
}
