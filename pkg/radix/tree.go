/*
Package radix implements a wrapper around an immutable radix tree data structure.
A radix tree (also known as a radix trie or compact prefix tree) is a space-optimized
tree structure that is particularly efficient for string or byte slice keys. It compresses
common prefixes to save space and enables fast lookups, insertions, and prefix-based searches.
*/
package radix

import (
	"bytes"
	"container/ring"
	"time"

	iradix "github.com/hashicorp/go-immutable-radix/v2"
)

/*
Tree wraps an immutable radix tree implementation from hashicorp/go-immutable-radix.
It stores byte slices as both keys and values, providing efficient prefix-based operations.
The immutable nature ensures thread-safety and enables persistent data structures.
*/
type Tree struct {
	root    *iradix.Tree[[]byte]
	updated bool
	perfs   *ring.Ring
	// Add persistence store
	persist *PersistentStore
}

/*
NewTree creates and returns a new empty Tree instance.
The underlying radix tree is initialized with no entries.
*/
func NewTree(persistDir string) (*Tree, error) {
	var persist *PersistentStore
	var err error

	if persistDir != "" {
		persist, err = NewPersistentStore(persistDir)
		if err != nil {
			return nil, err
		}
	}

	return &Tree{
		root:    iradix.New[[]byte](),
		perfs:   ring.New(10),
		persist: persist,
	}, nil
}

/*
Seek performs a prefix-based search in the tree, finding the first value whose key
is greater than or equal to the provided key in lexicographical order.
Returns the value and true if found, or nil and false if no such key exists.
*/
func (tree *Tree) Seek(key []byte) ([]byte, bool) {
	t := time.Now()

	it := tree.root.Root().Iterator()
	it.SeekLowerBound(key)

	for k, v, ok := it.Next(); ok; k, v, ok = it.Next() {
		if bytes.Compare(k, key) >= 0 {
			return v, true
		}
	}

	tree.perfs.Value = time.Since(t).Nanoseconds()
	tree.perfs = tree.perfs.Next()

	return nil, false
}

/*
Insert adds or updates a key-value pair in the tree.
Due to the immutable nature of the tree, this operation creates a new version
of the tree rather than modifying the existing one.
Returns the updated tree and a boolean indicating if the tree was modified.
*/
func (tree *Tree) Insert(key []byte, value []byte) (*Tree, bool) {
	t := time.Now()
	tree.root, _, tree.updated = tree.root.Insert(key, value)

	// Log to WAL if persistence is enabled
	if tree.persist != nil && tree.updated {
		if err := tree.persist.LogInsert(key, value); err != nil {
			// Log error but don't fail the operation
			// TODO: Add proper error handling/logging
			_ = err
		}
	}

	tree.perfs.Value = time.Since(t).Nanoseconds()
	tree.perfs = tree.perfs.Next()
	return tree, tree.updated
}

/*
Get retrieves the value associated with the given key.
Returns the value and true if the key exists, or nil and false if it doesn't.
*/
func (tree *Tree) Get(key []byte) ([]byte, bool) {
	t := time.Now()
	v, ok := tree.root.Get(key)
	tree.perfs.Value = time.Since(t).Nanoseconds()
	tree.perfs = tree.perfs.Next()
	return v, ok
}

/*
AVG returns the average performance of the tree in nanoseconds.
*/
func (tree *Tree) AVG() int64 {
	var sum int64

	tree.perfs.Do(func(v any) {
		sum += v.(int64)
	})

	return sum / int64(tree.perfs.Len())
}

// Close releases any resources held by the tree
func (tree *Tree) Close() error {
	if tree.persist != nil {
		return tree.persist.Close()
	}
	return nil
}
