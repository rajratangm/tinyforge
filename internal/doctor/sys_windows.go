//go:build windows

package doctor

import (
	"runtime"
	"syscall"
	"unsafe"
)

type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

// ReadSys reads CPU count, RAM (GlobalMemoryStatusEx) and free disk (GetDiskFreeSpaceExW) for workdir.
func ReadSys(workdir string) SysInfo {
	s := SysInfo{CPUs: runtime.NumCPU()}
	k32 := syscall.NewLazyDLL("kernel32.dll")

	var m memoryStatusEx
	m.Length = uint32(unsafe.Sizeof(m))
	if r, _, err := k32.NewProc("GlobalMemoryStatusEx").Call(uintptr(unsafe.Pointer(&m))); r == 0 {
		s.RAMError = "GlobalMemoryStatusEx failed: " + err.Error()
	} else {
		s.RAMTotalBytes, s.RAMAvailBytes = m.TotalPhys, m.AvailPhys
	}

	dir, err := syscall.UTF16PtrFromString(workdir)
	if err != nil {
		s.DiskError = err.Error()
		return s
	}
	var free, total, totalFree uint64
	r, _, e := k32.NewProc("GetDiskFreeSpaceExW").Call(uintptr(unsafe.Pointer(dir)),
		uintptr(unsafe.Pointer(&free)), uintptr(unsafe.Pointer(&total)), uintptr(unsafe.Pointer(&totalFree)))
	if r == 0 {
		s.DiskError = "GetDiskFreeSpaceExW failed: " + e.Error()
		return s
	}
	s.DiskFreeBytes, s.DiskKnown = free, true
	return s
}
