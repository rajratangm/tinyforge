//go:build linux

package doctor

import (
	"os"
	"runtime"
	"syscall"
)

// ReadSys reads CPU count, RAM (/proc/meminfo) and free disk (statfs) for workdir.
func ReadSys(workdir string) SysInfo {
	s := SysInfo{CPUs: runtime.NumCPU()}
	if b, err := os.ReadFile("/proc/meminfo"); err != nil {
		s.RAMError = err.Error()
	} else if total, avail, err := parseMeminfo(string(b)); err != nil {
		s.RAMError = err.Error()
	} else {
		s.RAMTotalBytes, s.RAMAvailBytes = total, avail
	}
	var st syscall.Statfs_t
	if err := syscall.Statfs(workdir, &st); err != nil {
		s.DiskError = err.Error()
		return s
	}
	s.DiskFreeBytes, s.DiskKnown = uint64(st.Bavail)*uint64(st.Bsize), true
	return s
}
