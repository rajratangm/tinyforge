package runner

import (
	"errors"
	"io"
	"os"
	"os/exec"
)

// Child is a running worker process. It is an interface so tests can use a fake and never start Python.
type Child interface {
	Stdout() io.Reader
	// Forward asks the child to stop gracefully (finish the step, write a checkpoint). See proc_*.go for platform details.
	Forward(sig os.Signal) error
	// Kill stops the child immediately; the worker cannot checkpoint.
	Kill() error
	// Wait returns the exit code once stdout has been read to EOF. A child killed by a signal reports -1.
	Wait() (int, error)
}

// Launcher starts a child process, wiring its stderr to the given writer.
type Launcher func(name string, args []string, stderr io.Writer) (Child, error)

type execChild struct {
	cmd *exec.Cmd
	out io.Reader
}

// ExecLauncher is the real Launcher, backed by os/exec.
func ExecLauncher(name string, args []string, stderr io.Writer) (Child, error) {
	cmd := exec.Command(name, args...)
	cmd.Stderr = stderr
	cmd.SysProcAttr = sysProcAttr()
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &execChild{cmd: cmd, out: out}, nil
}

func (c *execChild) Stdout() io.Reader           { return c.out }
func (c *execChild) Forward(sig os.Signal) error { return forward(c.cmd.Process, sig) }
func (c *execChild) Kill() error                 { return c.cmd.Process.Kill() }

func (c *execChild) Wait() (int, error) {
	err := c.cmd.Wait()
	if err == nil {
		return 0, nil
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return ee.ExitCode(), nil
	}
	return -1, err
}
