# LensDB - Image Description Tool

A Go tool that processes images in a folder using the Anthropic API to generate descriptions and stores them in a SQLite database.

## Installation

```bash
go mod download
go build
```

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
# Set environment variables
export ANTHROPIC_API_KEY=your-api-key-here
export OLLAMA_URL=http://192.168.1.100:11434

# Run the tool (uses environment variables)
./lensdb /path/to/images

# Or use command-line flags (override environment variables)
./lensdb /path/to/images -api-key your-api-key-here -ollama-url http://localhost:11434

# Specify custom database path and embedding model
./lensdb /path/to/images -db custom.db -embedding-model mxbai-embed-large
```

### Searching Images

Search your processed images using natural language queries:

```bash
# Basic search (uses environment variables if set)
./lensdb -search "sunset over mountains"

# Search with custom database and Ollama server via flags
./lensdb -search "people smiling" -db custom.db -ollama-url http://192.168.1.100:11434

# Or set via environment variables
export OLLAMA_URL=http://192.168.1.100:11434
export OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
./lensdb -search "dogs playing"
```

### Finding Similar Images

Find images similar to a specific image in your database:

```bash
# Find images similar to a specific image
./lensdb -similar /path/to/image.jpg

# With custom database
./lensdb -similar /path/to/image.jpg -db custom.db

# Note: The image must already be in the database
# Run lensdb on the folder containing the image first if it's not indexed yet
```

### Web Interface

Start a web server to search your images through a browser interface:

```bash
# Start web interface on default port 8080
./lensdb -web

# Use custom port
./lensdb -web -port 3000

# With custom database and Ollama configuration
./lensdb -web -port 3000 -db custom.db -ollama-url http://192.168.1.100:11434
```

Then open your browser to `http://localhost:8080` (or your specified port) to search interactively. The web interface includes "Find Similar" buttons for each search result.

## Flags

Command-line flags override environment variables:

- `-db`: Path to SQLite database file (default: `~/.lensdb.db`)
- `-api-key`: Anthropic API key (env: `ANTHROPIC_API_KEY`)
- `-ollama-url`: Ollama server URL (env: `OLLAMA_URL`, default: `http://localhost:11434`)
- `-embedding-model`: Ollama embedding model (env: `OLLAMA_EMBEDDING_MODEL`, default: `nomic-embed-text`)
- `-search`: Search query for semantic image search
- `-similar`: Find images similar to the specified image path
- `-web`: Start web interface for interactive searching
- `-port`: Port for web interface (default: `8080`)

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)

## Features

- **Web Interface**: Interactive browser-based search interface for exploring your image collection with real-time semantic search results and "Find Similar" functionality.
- **Semantic Search**: Search your image collection using natural language queries powered by Ollama embeddings and sqlite-vec. Find images based on their content, not just filenames.
- **Find Similar Images**: Discover images visually similar to a given image by comparing their embeddings. Works in both CLI and web interface modes.
- **Smart Duplicate Detection**: Images already in the database are automatically skipped, avoiding redundant API calls and saving costs. Re-run the tool on the same folder anytime to process only new images.
- **Automatic Image Resizing**: Large images are automatically resized to 1000px on their longest side before being sent to the API. This ensures images stay under the 5MB API limit while maintaining quality.
- **High-Quality Processing**: Uses Catmull-Rom interpolation for smooth, professional-looking resized images.
- **Format Preservation**: Resized images maintain their original format where possible (JPEG at 85% quality, PNG, GIF).
- **Remote Ollama Support**: Connect to Ollama running on any machine in your network for embeddings generation.

## Prerequisites

- **Ollama**: Install and run [Ollama](https://ollama.com/) for embeddings generation
- **Embedding Model**: Pull the embedding model (default is `nomic-embed-text`):
  ```bash
  ollama pull nomic-embed-text
  ```
- **Anthropic API Key**: Get your API key from [Anthropic Console](https://console.anthropic.com/)

## Environment Variables

You can configure LensDB using environment variables instead of command-line flags:

```bash
# Required for processing images
export ANTHROPIC_API_KEY=your-api-key-here

# Optional - Ollama configuration
export OLLAMA_URL=http://192.168.1.100:11434  # default: http://localhost:11434
export OLLAMA_EMBEDDING_MODEL=mxbai-embed-large  # default: nomic-embed-text
```

These can be set in a `.env` file when using `run.sh` with dotenvx.

## Database Schema

The tool creates a SQLite database with the following tables:

```sql
-- Main table for image metadata
CREATE TABLE image_descriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    foldername TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    embedding BLOB,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Virtual table for vector similarity search
CREATE VIRTUAL TABLE vec_descriptions USING vec0(
    embedding float[768]
);
```

## Example

### Processing Images

```bash
./lensdb ~/Pictures/vacation
```

This will:
1. Scan all images in the `~/Pictures/vacation` folder (including subdirectories)
2. Skip any images already processed (already in database)
3. Resize any large images to 1000px maximum on the longest side
4. Send each new image to the Anthropic API for description
5. Generate embeddings from descriptions using Ollama
6. Store the results and embeddings in `~/.lensdb.db`

Running the same command again will only process newly added images, skipping those already in the database.

### Searching Images

```bash
./lensdb -search "beach sunset"
```

This will:
1. Generate an embedding for your query using Ollama
2. Search the database for images with similar embeddings
3. Display the top 10 matching images with their paths and descriptions

## Querying the Database

```bash
sqlite3 ~/.lensdb.db "SELECT filename, description FROM image_descriptions;"
```

## Dependencies

LensDB uses the following Go packages:

- **github.com/mattn/go-sqlite3** (v1.14.32): SQLite database driver
- **github.com/asg017/sqlite-vec-go-bindings** (v0.1.6): SQLite vector search extension
- **golang.org/x/image** (v0.34.0): Image processing and format support (including WebP)

### System Requirements

- Go 1.24.0 or higher
- CGO enabled (required for SQLite compilation)
- Ollama running locally or on remote server
- Anthropic API key for image description generation
