//go:build !windows && !linux

package doctor

import "runtime"

// ReadSys is a stub on platforms forgectl does not target yet (Linux first, Windows as the dev box).
func ReadSys(workdir string) SysInfo {
	const why = "host probing is not implemented on this OS"
	return SysInfo{CPUs: runtime.NumCPU(), RAMError: why, DiskError: why}
}
