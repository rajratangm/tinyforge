//go:build !windows

package runner

import (
	"os"
	"syscall"
)

// The worker runs in its own process group so a terminal Ctrl-C reaches only forgectl, which then forwards a single
// SIGTERM. The worker's handler finishes the step, writes an atomic checkpoint and exits 75.
func sysProcAttr() *syscall.SysProcAttr { return &syscall.SysProcAttr{Setpgid: true} }

func forward(p *os.Process, _ os.Signal) error { return p.Signal(syscall.SIGTERM) }
