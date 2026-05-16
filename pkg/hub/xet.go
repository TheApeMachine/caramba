package hub

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type xetTokenPayload struct {
	AccessToken string `json:"accessToken"`
	ExpiresAt   int64  `json:"exp"`
	CASURL      string `json:"casUrl"`
}

type xetReconstruction struct {
	OffsetIntoFirstRange int64                     `json:"offset_into_first_range"`
	Terms                []xetTerm                 `json:"terms"`
	FetchInfo            map[string][]xetFetchInfo `json:"fetch_info"`
}

type xetTerm struct {
	Hash           string   `json:"hash"`
	UnpackedLength int64    `json:"unpacked_length"`
	Range          xetRange `json:"range"`
}

type xetFetchInfo struct {
	Range    xetRange `json:"range"`
	URL      string   `json:"url"`
	URLRange xetRange `json:"url_range"`
}

type xetRange struct {
	Start int64 `json:"start"`
	End   int64 `json:"end"`
}

func (client *Client) downloadXet(
	ctx context.Context,
	request DownloadRequest,
	probe remoteMetadata,
	tmpPath string,
) (remoteMetadata, error) {
	token, err := client.xetToken(ctx, request)

	if err != nil {
		return remoteMetadata{}, err
	}

	reconstruction, err := client.xetReconstruction(
		ctx,
		token,
		probe.XetHash,
	)

	if err != nil {
		return remoteMetadata{}, err
	}

	if err := os.MkdirAll(filepath.Dir(tmpPath), 0o755); err != nil {
		return remoteMetadata{}, fmt.Errorf("hub: mkdir temp: %w", err)
	}

	file, err := os.Create(tmpPath)

	if err != nil {
		return remoteMetadata{}, fmt.Errorf("hub: create temp: %w", err)
	}

	defer file.Close()

	chunkCache := make(map[string]map[int64][]byte)
	offset := reconstruction.OffsetIntoFirstRange

	for termIndex, term := range reconstruction.Terms {
		chunks, err := client.xetTermChunks(ctx, term, reconstruction, chunkCache)

		if err != nil {
			return remoteMetadata{}, err
		}

		var written int64

		for chunkIndex := term.Range.Start; chunkIndex < term.Range.End; chunkIndex++ {
			chunk, ok := chunks[chunkIndex]

			if !ok {
				return remoteMetadata{}, fmt.Errorf(
					"hub: xet chunk %d missing from %s",
					chunkIndex,
					term.Hash,
				)
			}

			if termIndex == 0 && offset > 0 {
				if offset >= int64(len(chunk)) {
					offset -= int64(len(chunk))
					continue
				}

				chunk = chunk[offset:]
				offset = 0
			}

			size, err := file.Write(chunk)

			if err != nil {
				return remoteMetadata{}, fmt.Errorf("hub: write xet file: %w", err)
			}

			written += int64(size)
		}

		if term.UnpackedLength > 0 && written > term.UnpackedLength {
			return remoteMetadata{}, fmt.Errorf(
				"hub: xet term %s wrote %d bytes, expected at most %d",
				term.Hash,
				written,
				term.UnpackedLength,
			)
		}
	}

	return remoteMetadata{
		ETag:    probe.ETag,
		XetHash: probe.XetHash,
		Size:    probe.Size,
	}, nil
}

func (client *Client) xetToken(
	ctx context.Context, request DownloadRequest,
) (xetTokenPayload, error) {
	apiPlural, err := request.RepoType.apiPlural()

	if err != nil {
		return xetTokenPayload{}, err
	}

	requestURL := fmt.Sprintf(
		"%s/api/%s/%s/xet-read-token/%s",
		strings.TrimRight(client.config.Endpoint, "/"),
		apiPlural,
		escapeRepoID(request.RepoID),
		url.PathEscape(request.Revision),
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)

	if err != nil {
		return xetTokenPayload{}, fmt.Errorf("hub: build xet token request: %w", err)
	}

	authorize(req, request.Token)

	resp, err := client.httpClient.Do(req)

	if err != nil {
		return xetTokenPayload{}, fmt.Errorf("hub: xet token: %w", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return xetTokenPayload{}, statusError("hub: xet token", resp)
	}

	var payload xetTokenPayload

	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return xetTokenPayload{}, fmt.Errorf("hub: decode xet token: %w", err)
	}

	if payload.AccessToken == "" || payload.CASURL == "" {
		return xetTokenPayload{}, fmt.Errorf("hub: xet token response missing accessToken or casUrl")
	}

	return payload, nil
}

func (client *Client) xetReconstruction(
	ctx context.Context,
	token xetTokenPayload,
	fileID string,
) (xetReconstruction, error) {
	requestURL := fmt.Sprintf(
		"%s/v1/reconstructions/%s",
		strings.TrimRight(token.CASURL, "/"),
		fileID,
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)

	if err != nil {
		return xetReconstruction{}, fmt.Errorf("hub: build xet reconstruction request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+token.AccessToken)

	resp, err := client.httpClient.Do(req)

	if err != nil {
		return xetReconstruction{}, fmt.Errorf("hub: xet reconstruction: %w", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return xetReconstruction{}, statusError("hub: xet reconstruction", resp)
	}

	var reconstruction xetReconstruction

	if err := json.NewDecoder(resp.Body).Decode(&reconstruction); err != nil {
		return xetReconstruction{}, fmt.Errorf("hub: decode xet reconstruction: %w", err)
	}

	return reconstruction, nil
}

func (client *Client) xetTermChunks(
	ctx context.Context,
	term xetTerm,
	reconstruction xetReconstruction,
	chunkCache map[string]map[int64][]byte,
) (map[int64][]byte, error) {
	fetch, err := matchingFetchInfo(term, reconstruction.FetchInfo[term.Hash])

	if err != nil {
		return nil, err
	}

	cacheKey := term.Hash + ":" +
		strconv.FormatInt(fetch.Range.Start, 10) + ":" +
		strconv.FormatInt(fetch.Range.End, 10) + ":" +
		strconv.FormatInt(fetch.URLRange.Start, 10) + ":" +
		strconv.FormatInt(fetch.URLRange.End, 10)

	if chunks, ok := chunkCache[cacheKey]; ok {
		return chunks, nil
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fetch.URL, nil)

	if err != nil {
		return nil, fmt.Errorf("hub: build xet xorb request: %w", err)
	}

	req.Header.Set(
		"Range",
		fmt.Sprintf("bytes=%d-%d", fetch.URLRange.Start, fetch.URLRange.End),
	)

	resp, err := client.httpClient.Do(req)

	if err != nil {
		return nil, fmt.Errorf("hub: xet xorb fetch: %w", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return nil, statusError("hub: xet xorb fetch", resp)
	}

	data, err := io.ReadAll(resp.Body)

	if err != nil {
		return nil, fmt.Errorf("hub: read xet xorb: %w", err)
	}

	chunks, err := DecodeXorbRange(data, fetch.Range.Start, fetch.Range.End)

	if err != nil {
		return nil, err
	}

	chunkCache[cacheKey] = chunks

	return chunks, nil
}

func matchingFetchInfo(term xetTerm, fetches []xetFetchInfo) (xetFetchInfo, error) {
	for _, fetch := range fetches {
		if fetch.Range.Start <= term.Range.Start && fetch.Range.End >= term.Range.End {
			return fetch, nil
		}
	}

	return xetFetchInfo{}, fmt.Errorf("hub: xet reconstruction has no fetch info for %s [%d,%d)", term.Hash, term.Range.Start, term.Range.End)
}

/*
DecodeXorbRange deserializes a Xet xorb byte range into chunk-indexed raw data.
*/
func DecodeXorbRange(data []byte, start, end int64) (map[int64][]byte, error) {
	reader := bytes.NewReader(data)
	chunks := make(map[int64][]byte, end-start)

	for chunkIndex := start; chunkIndex < end; chunkIndex++ {
		header := make([]byte, 8)

		if _, err := io.ReadFull(reader, header); err != nil {
			return nil, fmt.Errorf("hub: xorb chunk %d header: %w", chunkIndex, err)
		}

		if header[0] != 0 {
			return nil, fmt.Errorf("hub: xorb chunk %d version %d is unsupported", chunkIndex, header[0])
		}

		compressedSize := threeByteLE(header[1:4])
		compression := header[4]
		uncompressedSize := threeByteLE(header[5:8])

		compressed := make([]byte, compressedSize)

		if _, err := io.ReadFull(reader, compressed); err != nil {
			return nil, fmt.Errorf("hub: xorb chunk %d payload: %w", chunkIndex, err)
		}

		chunk, err := decodeXorbChunk(compressed, compression, uncompressedSize)

		if err != nil {
			return nil, fmt.Errorf("hub: xorb chunk %d: %w", chunkIndex, err)
		}

		chunks[chunkIndex] = chunk
	}

	return chunks, nil
}

func decodeXorbChunk(
	compressed []byte, compression byte, uncompressedSize int,
) ([]byte, error) {
	switch compression {
	case 0:
		if len(compressed) != uncompressedSize {
			return nil, fmt.Errorf("uncompressed size %d != %d", len(compressed), uncompressedSize)
		}

		return compressed, nil
	case 1:
		return decodeLZ4Payload(compressed, uncompressedSize)
	case 2:
		grouped, err := decodeLZ4Payload(compressed, uncompressedSize)

		if err != nil {
			return nil, err
		}

		return ungroup4(grouped), nil
	default:
		return nil, fmt.Errorf("compression type %d is unsupported", compression)
	}
}

func threeByteLE(data []byte) int {
	return int(data[0]) | int(data[1])<<8 | int(data[2])<<16
}
