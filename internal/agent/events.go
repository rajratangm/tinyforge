package agent

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

const (
	ringMaxLines   = 1000     // lines kept in memory per active job (what a late follower can replay)
	ringMaxBytes   = 1 << 20  // ...and at most this many bytes
	maxEventLine   = 64 << 10 // a longer worker line is replaced by a marker, never stored whole
	maxEventsFile  = 16 << 20 // after this the file stops growing (one marker line says so)
	tailReadBytes  = 4 << 20  // how much of the file end is read to rebuild a tail
	keepaliveEvery = 20 * time.Second
	writeDeadline  = 30 * time.Second
)

// eventLog holds a job's worker events: a bounded in-memory ring for live followers and an append-only file that
// survives restarts. Lines are the worker's own JSON lines, verbatim; the agent adds a few lifecycle lines
// ("event":"job_started" / "job_end" / "job_requeued") that consumers ignore if they do not know them.
type eventLog struct {
	mu        sync.Mutex
	path      string // "" = memory only. The file is opened per write and never held open, so nothing leaks.
	fileBytes int64
	fileFull  bool
	ring      [][]byte
	ringBytes int
	first     uint64 // sequence number of ring[0]
	next      uint64 // sequence number the next line will get
	notify    chan struct{}
	done      bool
}

// openEventLog opens (or creates) the file at path and seeds the ring from its tail. path "" = memory only.
func openEventLog(path string) *eventLog {
	l := &eventLog{notify: make(chan struct{}), path: path}
	if path == "" {
		return l
	}
	for _, line := range readTailFile(path, ringMaxLines) {
		l.pushRing(line)
	}
	if st, err := os.Stat(path); err == nil {
		l.fileBytes = st.Size()
	}
	return l
}

func (l *eventLog) pushRing(line []byte) {
	l.ring = append(l.ring, line)
	l.ringBytes += len(line)
	l.next++
	for len(l.ring) > 1 && (len(l.ring) > ringMaxLines || l.ringBytes > ringMaxBytes) {
		l.ringBytes -= len(l.ring[0])
		l.ring = l.ring[1:]
		l.first++
	}
}

func (l *eventLog) broadcast() {
	close(l.notify)
	l.notify = make(chan struct{})
}

// append stores one line (a trailing newline is trimmed; empty lines are dropped).
func (l *eventLog) append(line []byte) {
	line = bytes.TrimRight(line, "\r\n")
	if len(line) == 0 {
		return
	}
	if len(line) > maxEventLine {
		line = []byte(`{"event":"truncated","bytes":` + strconv.Itoa(len(line)) + `}`)
	} else {
		line = append([]byte(nil), line...)
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.done {
		return
	}
	l.pushRing(line)
	if l.path != "" && !l.fileFull {
		if l.fileBytes+int64(len(line))+1 > maxEventsFile {
			l.writeFile([]byte(`{"event":"events_truncated","note":"event file reached its size cap; later events are live-only"}` + "\n"))
			l.fileFull = true
		} else {
			l.fileBytes += int64(l.writeFile(append(append([]byte(nil), line...), '\n')))
		}
	}
	l.broadcast()
}

// writeFile appends to the event file (opened and closed here) and returns the bytes written. A failure only loses
// persistence for that line: the live stream is unaffected.
func (l *eventLog) writeFile(b []byte) int {
	f, err := os.OpenFile(l.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return 0
	}
	defer f.Close()
	n, _ := f.Write(b)
	return n
}

// appendEvent adds an agent-originated lifecycle line.
func (l *eventLog) appendEvent(kind string, fields map[string]any) {
	m := map[string]any{"event": kind, "t": float64(time.Now().UnixNano()) / 1e9, "source": "agent"}
	for k, v := range fields {
		m[k] = v
	}
	b, err := json.Marshal(m)
	if err == nil {
		l.append(b)
	}
}

// finish marks the log complete: followers drain what is left and stop.
func (l *eventLog) finish() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.done {
		return
	}
	l.done = true
	l.broadcast()
}

// tail returns the last n lines and the sequence number to continue from.
func (l *eventLog) tail(n int) ([][]byte, uint64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if n > len(l.ring) {
		n = len(l.ring)
	}
	return append([][]byte(nil), l.ring[len(l.ring)-n:]...), l.next
}

// read returns every line with sequence >= after. gap is true if some were already evicted from the ring.
func (l *eventLog) read(after uint64) (lines [][]byte, next uint64, gap, done bool, wait <-chan struct{}) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if after < l.first {
		after, gap = l.first, true
	}
	if after < l.next {
		lines = append(lines, l.ring[after-l.first:]...)
	}
	return lines, l.next, gap, l.done, l.notify
}

// readTailFile returns the last n non-empty lines of the file at path, reading at most tailReadBytes from its end.
func readTailFile(path string, n int) [][]byte {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil || st.Size() == 0 {
		return nil
	}
	size, off := st.Size(), int64(0)
	if size > tailReadBytes {
		off = size - tailReadBytes
	}
	buf := make([]byte, size-off)
	if _, err := f.ReadAt(buf, off); err != nil && err != io.EOF {
		return nil
	}
	parts := bytes.Split(buf, []byte("\n"))
	if off > 0 && len(parts) > 0 {
		parts = parts[1:] // the first line is probably cut in half
	}
	var lines [][]byte
	for _, p := range parts {
		if p = bytes.TrimRight(p, "\r"); len(p) > 0 {
			lines = append(lines, append([]byte(nil), p...))
		}
	}
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return lines
}

// lineWriter turns the runner's stdout writes into event lines. The runner writes one full line per Write, but
// this splits defensively in case a write ever carries several lines or a partial one. Only JSON objects are kept:
// the worker contract puts human text on stderr, and the NDJSON endpoint must never emit a non-JSON line.
type lineWriter struct {
	log *eventLog
	buf []byte
}

func (w *lineWriter) Write(p []byte) (int, error) {
	w.buf = append(w.buf, p...)
	for {
		i := bytes.IndexByte(w.buf, '\n')
		if i < 0 {
			break
		}
		w.emit(w.buf[:i])
		w.buf = w.buf[i+1:]
	}
	if len(w.buf) > 4<<20 { // a runaway line with no newline: do not grow without bound
		w.emit(w.buf)
		w.buf = nil
	}
	return len(p), nil
}

func (w *lineWriter) flush() {
	if len(w.buf) > 0 {
		w.emit(w.buf)
		w.buf = nil
	}
}

func (w *lineWriter) emit(line []byte) {
	if t := bytes.TrimSpace(line); len(t) > 1 && t[0] == '{' && json.Valid(t) {
		w.log.append(t)
	}
}

// jobEvents streams a job's events as NDJSON. It first replays up to ?tail=N recent lines (default 200, max 1000),
// then, for a job that is still active and unless ?follow=0, keeps streaming until the job reaches a terminal state,
// the client disconnects or the agent shuts down. A finished job is served from the persisted file.
func (s *Server) jobEvents(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !validID(id) {
		writeJSON(w, http.StatusNotFound, errBody("no such job"))
		return
	}
	_, live, ok := s.jobs.getWithEvents(id)
	if !ok {
		writeJSON(w, http.StatusNotFound, errBody("no such job"))
		return
	}
	tailN := 200
	if v := r.URL.Query().Get("tail"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 || n > ringMaxLines {
			writeJSON(w, http.StatusBadRequest, errBody("tail must be between 1 and "+strconv.Itoa(ringMaxLines)))
			return
		}
		tailN = n
	}
	follow := true
	if v := r.URL.Query().Get("follow"); v == "0" || v == "false" {
		follow = false
	}

	h := w.Header()
	h.Set("Content-Type", "application/x-ndjson")
	h.Set("Cache-Control", "no-store")
	h.Set("X-Accel-Buffering", "no") // do not let a reverse proxy buffer the stream
	w.WriteHeader(http.StatusOK)
	rc := http.NewResponseController(w)
	write := func(lines ...[]byte) bool {
		// The server's WriteTimeout would cut a long stream: push the deadline out before every chunk.
		_ = rc.SetWriteDeadline(time.Now().Add(writeDeadline))
		for _, l := range lines {
			if _, err := w.Write(append(append([]byte(nil), l...), '\n')); err != nil {
				return false
			}
		}
		_ = rc.Flush() // best effort: a writer without flush support just buffers
		return true
	}

	if live == nil { // finished: served from the persisted file
		write(readTailFile(s.jobs.eventsPath(id), tailN)...)
		return
	}
	lines, after := live.tail(tailN)
	if !write(lines...) || !follow {
		return
	}
	keepalive := time.NewTicker(keepaliveEvery)
	defer keepalive.Stop()
	for {
		ls, next, gap, done, wait := live.read(after)
		if gap && !write([]byte(`{"event":"gap","source":"agent","note":"some events were dropped from the live buffer"}`)) {
			return
		}
		if len(ls) > 0 && !write(ls...) {
			return
		}
		after = next
		if done {
			return
		}
		select {
		case <-wait:
		case <-r.Context().Done():
			return
		case <-s.closing:
			return
		case <-keepalive.C:
			if !write([]byte(`{"event":"keepalive","source":"agent"}`)) {
				return
			}
		}
	}
}
