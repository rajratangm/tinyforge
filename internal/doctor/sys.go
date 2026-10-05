package doctor

import (
	"fmt"
	"strconv"
	"strings"
)

// SysInfo is the host summary. *Error fields carry a reason when a value could not be read.
type SysInfo struct {
	OS            string `json:"os"`
	Arch          string `json:"arch"`
	CPUs          int    `json:"cpus"`
	RAMTotalBytes uint64 `json:"ram_total_bytes,omitempty"`
	RAMAvailBytes uint64 `json:"ram_available_bytes,omitempty"`
	RAMError      string `json:"ram_error,omitempty"`
	DiskFreeBytes uint64 `json:"disk_free_bytes,omitempty"`
	DiskKnown     bool   `json:"disk_known"`
	DiskError     string `json:"disk_error,omitempty"`
}

// parseMeminfo extracts MemTotal and MemAvailable (bytes) from /proc/meminfo text.
func parseMeminfo(s string) (total, avail uint64, err error) {
	var haveT, haveA bool
	for _, line := range strings.Split(s, "\n") {
		key, rest, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			continue
		}
		kb, perr := strconv.ParseUint(fields[0], 10, 64)
		if perr != nil {
			continue
		}
		switch key {
		case "MemTotal":
			total, haveT = kb*1024, true
		case "MemAvailable":
			avail, haveA = kb*1024, true
		}
	}
	if !haveT || !haveA {
		return 0, 0, fmt.Errorf("MemTotal/MemAvailable not found in meminfo")
	}
	return total, avail, nil
}
