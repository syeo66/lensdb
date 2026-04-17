# LensDB - Image Description Tool

A Go tool that processes images in a folder, generates descriptions using an Ollama vision model (or optionally the Anthropic API), creates semantic embeddings using Ollama, and stores them in a SQLite database with vector and full-text search capabilities.

## Installation

```bash
go mod download
make
```

> **Note:** Build via `make` (or `go build -tags fts5`) to enable full-text search. A plain `go build` omits fts5 support.

## Usage

### Processing Images

#### Using run.sh (with dotenvx)

```bash
# Run with run.sh (reads API key from .env file via dotenvx)
./run.sh /path/to/images

# Specify custom database path
./run.sh /path/to/images custom.db
```

#### Direct usage

```bash
# Run with default Ollama vision model (qwen3-vl:8b) — no Anthropic key needed
./lensdb /path/to/images

# Use a different Ollama vision model
./lensdb /path/to/images -vision-model llava:13b

# Use Anthropic Claude instead of Ollama for descriptions
export ANTHROPIC_API_KEY=your-api-key-here
./lensdb /path/to/images -use-anthropic

# With remote Ollama and custom database
./lensdb /path/to/images -ollama-url http://192.168.1.100:11434 -db custom.db

# Specify custom embedding model
./lensdb /path/to/images -embedding-model nomic-embed-text
```

### Searching Images

Search your processed images using natural language queries:

```bash
# Basic search
./lensdb -search "sunset over mountains"

# Search with custom database and Ollama server
./lensdb -search "people smiling" -db custom.db -ollama-url http://192.168.1.100:11434

# Or set via environment variables
export OLLAMA_URL=http://192.168.1.100:11434
./lensdb -search "dogs playing"
```

Search uses hybrid ranking: vector similarity and full-text search results are merged using Reciprocal Rank Fusion, so both semantic and keyword matches surface.

### Finding Similar Images

Find images similar to a specific image already in your database:

```bash
./lensdb -similar /path/to/image.jpg

# With custom database
./lensdb -similar /path/to/image.jpg -db custom.db
```

The image must already be indexed. Run lensdb on its folder first if it isn't.

### Web Interface

```bash
# Start web interface on default port 8080
./lensdb -web

# Use custom port
./lensdb -web -port 3000

# With custom database and Ollama configuration
./lensdb -web -port 3000 -db custom.db -ollama-url http://192.168.1.100:11434
```

Open `http://localhost:8080` (or your specified port) to search interactively. Each result has a "Find Similar" button.

### Reindexing

Rebuild the vector and full-text search indexes from the descriptions already stored in the database. Useful after switching embedding models or when the indexes become out of sync:

```bash
./lensdb -reindex

# With a custom database or Ollama server
./lensdb -reindex -db custom.db -ollama-url http://192.168.1.100:11434
```

> **Note:** If you change the embedding model, the vector dimensions may change. LensDB will automatically drop and recreate the vec table on startup, then prompt you to run `-reindex`.

### Cleanup

Remove database entries for image files that no longer exist on disk. Useful after moving, renaming, or deleting images:

```bash
./lensdb -cleanup

# With a custom database
./lensdb -cleanup -db custom.db
```

Displays a live `[n/total]` progress counter while checking, prints each removed path, and finishes with a summary.

## Flags

Command-line flags override environment variables:

- `-db`: Path to SQLite database file (default: `~/.lensdb.db`)
- `-ollama-url`: Ollama server URL (env: `OLLAMA_URL`, default: `http://localhost:11434`)
- `-embedding-model`: Ollama embedding model (env: `OLLAMA_EMBEDDING_MODEL`, default: `bge-m3`)
- `-vision-model`: Ollama vision model for image descriptions (env: `OLLAMA_VISION_MODEL`, default: `qwen3-vl:8b`)
- `-use-anthropic`: Use Anthropic Claude for image descriptions instead of Ollama
- `-api-key`: Anthropic API key (env: `ANTHROPIC_API_KEY`, only needed with `-use-anthropic`)
- `-search`: Search query for semantic image search
- `-similar`: Find images similar to the specified image path
- `-web`: Start web interface for interactive searching
- `-port`: Port for web interface (default: `8080`)
- `-reindex`: Rebuild vector and full-text search indexes from existing descriptions
- `-cleanup`: Remove database entries for image files that no longer exist on disk

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)

## Features

- **Ollama-First Descriptions**: Image descriptions are generated locally using any Ollama vision model (default: `qwen3-vl:8b`) — no Anthropic API key required.
- **Anthropic Claude Option**: Use Claude Sonnet via `-use-anthropic` for cloud-based descriptions.
- **Hybrid Search**: Combines vector similarity (semantic) and FTS5 full-text search via Reciprocal Rank Fusion for better results than either alone.
- **Web Interface**: Interactive browser-based search with real-time results and "Find Similar" buttons.
- **Find Similar Images**: Discover visually similar images by comparing embeddings — available in both CLI and web modes.
- **Smart Duplicate Detection**: Images already in the database are automatically skipped.
- **Reindex**: Rebuild indexes at any time without re-describing images — useful after switching embedding models.
- **Cleanup**: Remove stale database entries for images that have been moved or deleted, with a live progress counter.
- **Automatic Image Resizing**: Images are resized to 1000px on the longest side before being sent to the vision model, using Catmull-Rom interpolation for quality.
- **Remote Ollama Support**: Connect to Ollama on any machine in your network.

## Prerequisites

- **Ollama**: Install and run [Ollama](https://ollama.com/)
- **Vision Model**: Pull the default vision model:
  ```bash
  ollama pull qwen3-vl:8b
  ```
- **Embedding Model**: Pull the default embedding model:
  ```bash
  ollama pull bge-m3
  ```
- **Anthropic API Key** *(optional)*: Only required with `-use-anthropic`. Get your key from [Anthropic Console](https://console.anthropic.com/).

## Environment Variables

```bash
export OLLAMA_URL=http://192.168.1.100:11434   # default: http://localhost:11434
export OLLAMA_VISION_MODEL=llava:13b           # default: qwen3-vl:8b
export OLLAMA_EMBEDDING_MODEL=nomic-embed-text # default: bge-m3

# Only needed when using -use-anthropic
export ANTHROPIC_API_KEY=your-api-key-here
```

These can be set in a `.env` file when using `run.sh` with dotenvx.

## Database Schema

```sql
-- Main table for image metadata
CREATE TABLE image_descriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    foldername TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    embedding BLOB,
    thumbnail BLOB,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Virtual table for vector similarity search (1024-dim, matches bge-m3)
CREATE VIRTUAL TABLE vec_descriptions USING vec0(
    embedding float[1024]
);

-- Virtual table for full-text search (requires -tags fts5 build)
CREATE VIRTUAL TABLE fts_descriptions USING fts5(description);
```

The vec table dimensions are controlled by the `embeddingDimensions` constant in `main.go`. If the on-disk table has different dimensions (e.g. after switching from `nomic-embed-text` at 768-dim), LensDB drops and recreates it automatically on startup and asks you to run `-reindex`.

## Example

### Processing Images

```bash
./lensdb ~/Pictures/vacation
```

This will:
1. Scan all images recursively in `~/Pictures/vacation`
2. Skip images already in the database
3. Resize large images to 1000px maximum on the longest side
4. Describe each new image with `qwen3-vl:8b` via Ollama
5. Generate 1024-dim embeddings using `bge-m3`
6. Store results in `~/.lensdb.db`

Run the same command again at any time — only new images are processed.

### Searching Images

```bash
./lensdb -search "beach sunset"
```

This generates a query embedding, runs vector + full-text search, merges the results, and returns the top 10 matches with paths and descriptions.

## Querying the Database Directly

```bash
sqlite3 ~/.lensdb.db "SELECT filename, description FROM image_descriptions;"
```

## Dependencies

- **github.com/mattn/go-sqlite3** (v1.14.34): SQLite database driver (CGO)
- **github.com/asg017/sqlite-vec-go-bindings** (v0.1.6): SQLite vector search extension
- **golang.org/x/image** (v0.36.0): Image processing and WebP support

### System Requirements

- Go 1.24.0 or higher
- CGO enabled (required for SQLite compilation)
- Ollama running locally or on a remote server
- Anthropic API key only required with `-use-anthropic`
