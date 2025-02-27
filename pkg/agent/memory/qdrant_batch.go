/*
Package memory provides memory-related functionality for the agent system.
This file contains batch operation implementations for QDrant vector store.
*/
package memory

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/qdrant/go-client/qdrant"
)

// BatchStoreVectors stores multiple vectors in a single operation for QDrant.
// This is more efficient than storing vectors one by one.
func (q *QDrantStore) BatchStoreVectors(
	ctx context.Context,
	ids []string,
	vectors [][]float32,
	payloads []map[string]interface{},
) error {
	if len(ids) == 0 || len(vectors) == 0 || len(payloads) == 0 {
		return fmt.Errorf("no vectors to store")
	}

	if len(ids) != len(vectors) || len(ids) != len(payloads) {
		return fmt.Errorf("mismatch in number of ids, vectors, and payloads")
	}

	// Process in chunks to avoid too large requests
	chunkSize := 100
	for i := 0; i < len(ids); i += chunkSize {
		end := i + chunkSize
		if end > len(ids) {
			end = len(ids)
		}

		points := make([]*qdrant.PointStruct, 0, end-i)

		// Create points for this chunk
		for j := i; j < end; j++ {
			// Process payload to handle special types
			processedPayload := make(map[string]interface{})
			for k, v := range payloads[j] {
				// Convert time.Time to RFC3339 string format
				if timeVal, ok := v.(time.Time); ok {
					processedPayload[k] = timeVal.Format(time.RFC3339)
				} else if strMap, ok := v.(map[string]string); ok {
					// Convert map[string]string to map[string]interface{}
					interfaceMap := make(map[string]interface{})
					for sk, sv := range strMap {
						interfaceMap[sk] = sv
					}
					processedPayload[k] = interfaceMap
				} else {
					processedPayload[k] = v
				}
			}

			// Convert payload to the Qdrant Value map
			qdrantPayload := qdrant.NewValueMap(processedPayload)

			// Create point with vectors
			points = append(points, &qdrant.PointStruct{
				Id:      qdrant.NewIDUUID(ids[j]),
				Vectors: qdrant.NewVectors(vectors[j]...),
				Payload: qdrantPayload,
			})
		}

		// Upsert the points
		_, err := q.client.Upsert(ctx, &qdrant.UpsertPoints{
			CollectionName: q.collection,
			Points:         points,
		})

		if err != nil {
			return fmt.Errorf("failed to batch store vectors in QDrant: %w", err)
		}
	}

	return nil
}

// BatchSearch performs multiple searches efficiently.
// For smaller batches, it runs searches in parallel.
// For larger batches, it processes sequentially to avoid overwhelming the server.
func (q *QDrantStore) BatchSearch(
	ctx context.Context,
	vectors [][]float32,
	limit int,
	filters []map[string]interface{},
) ([][]SearchResult, error) {
	if len(vectors) == 0 {
		return nil, fmt.Errorf("no vectors to search")
	}

	if limit <= 0 {
		limit = 10
	}

	results := make([][]SearchResult, len(vectors))

	// For smaller batches, use parallel processing
	if len(vectors) <= 5 {
		var wg sync.WaitGroup
		var mu sync.Mutex
		errs := make([]error, 0)

		for i, vector := range vectors {
			wg.Add(1)
			go func(idx int, vec []float32) {
				defer wg.Done()

				// Get filter for this query if available
				var filter map[string]interface{}
				if idx < len(filters) {
					filter = filters[idx]
				}

				// Perform search
				searchResults, err := q.Search(ctx, vec, limit, filter)
				if err != nil {
					mu.Lock()
					errs = append(errs, fmt.Errorf("search %d failed: %w", idx, err))
					mu.Unlock()
					return
				}

				mu.Lock()
				results[idx] = searchResults
				mu.Unlock()
			}(i, vector)
		}

		wg.Wait()

		if len(errs) > 0 {
			return results, fmt.Errorf("some batch searches failed: %v", errs)
		}
	} else {
		// For larger batches, process sequentially
		for i, vector := range vectors {
			// Get filter for this query if available
			var filter map[string]interface{}
			if i < len(filters) {
				filter = filters[i]
			}

			// Perform search
			searchResults, err := q.Search(ctx, vector, limit, filter)
			if err != nil {
				return results, fmt.Errorf("search %d failed: %w", i, err)
			}

			results[i] = searchResults
		}
	}

	return results, nil
}
