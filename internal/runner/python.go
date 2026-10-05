package runner

import (
	"errors"
	"path/filepath"
)

// FindPython picks the interpreter for the worker. Order: explicit (--python), FORGECTL_PYTHON, a project venv found
// by walking up from startDir (.venv/Scripts/python.exe or .venv/bin/python), then `python` on PATH.
// All environment access is injected so the order can be tested without touching the machine.
func FindPython(explicit string, getenv func(string) string, startDir string,
	exists func(string) bool, lookPath func(string) (string, error)) (path, source string, err error) {
	if explicit != "" {
		return explicit, "--python", nil
	}
	if v := getenv("FORGECTL_PYTHON"); v != "" {
		return v, "FORGECTL_PYTHON", nil
	}
	dir := startDir
	for {
		for _, rel := range []string{filepath.Join(".venv", "Scripts", "python.exe"), filepath.Join(".venv", "bin", "python")} {
			if p := filepath.Join(dir, rel); exists(p) {
				return p, "project venv", nil
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	if p, lerr := lookPath("python"); lerr == nil {
		return p, "PATH", nil
	}
	return "", "", errors.New("no Python found: pass --python, set FORGECTL_PYTHON, create a .venv, or put python on PATH")
}
