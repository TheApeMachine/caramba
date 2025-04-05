/*
Package radix implements persistence functionality for the radix tree.
The persistence layer uses a Write-Ahead Log (WAL) to ensure data durability
and provides mechanisms for recovery in case of failures.
*/
package radix

import (
	"bufio"
	"encoding/binary"
	"os"
	"path/filepath"
	"sync"
)

/*
Operation types for the WAL. These define the possible operations that can be
recorded in the write-ahead log for persistence and recovery.
*/
const (
	opInsert byte = iota
	opDelete
)

/*
WALEntry represents a single write-ahead log entry. Each entry contains the
operation type and the associated key-value pair, allowing for replay during
recovery operations.
*/
type WALEntry struct {
	Op    byte
	Key   []byte
	Value []byte
}

/*
PersistentStore handles the persistence layer for the radix tree.
It manages write-ahead logging and provides mechanisms for durable storage
and recovery of tree data.
*/
type PersistentStore struct {
	walFile    *os.File
	walWriter  *bufio.Writer
	walPath    string
	dataPath   string
	writeMutex sync.Mutex
	syncChan   chan struct{}
}

/*
NewPersistentStore creates a new persistent store instance.
It initializes the WAL file and sets up background syncing to ensure
data durability. The store will create necessary directories if they
don't exist.
*/
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

/*
LogInsert asynchronously logs an insert operation to the WAL.
It writes the operation type, key, and value to the WAL buffer and
signals the background sync goroutine to flush to disk.
*/
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

/*
backgroundSync periodically flushes the WAL to disk.
It listens on the sync channel and ensures that buffered writes are
persisted to stable storage.
*/
func (ps *PersistentStore) backgroundSync() {
	for range ps.syncChan {
		ps.writeMutex.Lock()
		ps.walWriter.Flush()
		ps.walFile.Sync()
		ps.writeMutex.Unlock()
	}
}

/*
Close closes the persistent store, ensuring all buffered data is
written to disk and resources are properly released.
*/
func (ps *PersistentStore) Close() error {
	close(ps.syncChan)
	ps.writeMutex.Lock()
	defer ps.writeMutex.Unlock()

	if err := ps.walWriter.Flush(); err != nil {
		return err
	}
	return ps.walFile.Close()
}
