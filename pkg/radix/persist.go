package radix

import (
	"bufio"
	"encoding/binary"
	"os"
	"path/filepath"
	"sync"
)

// Operation types for the WAL
const (
	opInsert byte = iota
	opDelete
)

// WALEntry represents a single write-ahead log entry
type WALEntry struct {
	Op    byte
	Key   []byte
	Value []byte
}

// PersistentStore handles the persistence layer for the radix tree
type PersistentStore struct {
	walFile    *os.File
	walWriter  *bufio.Writer
	walPath    string
	dataPath   string
	writeMutex sync.Mutex
	syncChan   chan struct{}
}

// NewPersistentStore creates a new persistent store instance
func NewPersistentStore(dir string) (*PersistentStore, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	walPath := filepath.Join(dir, "wal.log")
	dataPath := filepath.Join(dir, "data.snap")

	walFile, err := os.OpenFile(walPath, os.O_CREATE|os.O_APPEND|os.O_RDWR, 0644)
	if err != nil {
		return nil, err
	}

	ps := &PersistentStore{
		walFile:   walFile,
		walWriter: bufio.NewWriter(walFile),
		walPath:   walPath,
		dataPath:  dataPath,
		syncChan:  make(chan struct{}, 100),
	}

	// Start background syncing
	go ps.backgroundSync()

	return ps, nil
}

// LogInsert asynchronously logs an insert operation to the WAL
func (ps *PersistentStore) LogInsert(key, value []byte) error {
	ps.writeMutex.Lock()
	defer ps.writeMutex.Unlock()

	// Write operation type
	if err := ps.walWriter.WriteByte(opInsert); err != nil {
		return err
	}

	// Write key length and key
	if err := binary.Write(ps.walWriter, binary.LittleEndian, uint32(len(key))); err != nil {
		return err
	}
	if _, err := ps.walWriter.Write(key); err != nil {
		return err
	}

	// Write value length and value
	if err := binary.Write(ps.walWriter, binary.LittleEndian, uint32(len(value))); err != nil {
		return err
	}
	if _, err := ps.walWriter.Write(value); err != nil {
		return err
	}

	// Signal for background sync
	select {
	case ps.syncChan <- struct{}{}:
	default:
	}

	return nil
}

// backgroundSync periodically flushes the WAL to disk
func (ps *PersistentStore) backgroundSync() {
	for range ps.syncChan {
		ps.writeMutex.Lock()
		ps.walWriter.Flush()
		ps.walFile.Sync()
		ps.writeMutex.Unlock()
	}
}

// Close closes the persistent store
func (ps *PersistentStore) Close() error {
	close(ps.syncChan)
	ps.writeMutex.Lock()
	defer ps.writeMutex.Unlock()

	if err := ps.walWriter.Flush(); err != nil {
		return err
	}
	return ps.walFile.Close()
}
