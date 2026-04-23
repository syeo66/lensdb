package main

import (
	"bytes"
	"database/sql"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"flag"
	"fmt"
	"html/template"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"

	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	_ "github.com/mattn/go-sqlite3"
	"golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
)

const anthropicAPIURL = "https://api.anthropic.com/v1/messages"

// embeddingDimensions must match the model in use.
// bge-m3 (default) produces 1024-dim vectors.
const embeddingDimensions = 1024

type ImageDescription struct {
	Filename    string
	Foldername  string
	Path        string
	Description string
}

type SearchResult struct {
	Filename    string
	Foldername  string
	Path        string
	Description string
	Distance    float64
	Thumbnail   string // base64-encoded JPEG for data URL
}

type AnthropicRequest struct {
	Model     string    `json:"model"`
	MaxTokens int       `json:"max_tokens"`
	Messages  []Message `json:"messages"`
}

type Message struct {
	Role    string        `json:"role"`
	Content []ContentItem `json:"content"`
}

type ContentItem struct {
	Type   string       `json:"type"`
	Text   string       `json:"text,omitempty"`
	Source *ImageSource `json:"source,omitempty"`
}

type ImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type AnthropicResponse struct {
	Content []struct {
		Text string `json:"text"`
	} `json:"content"`
}

type OllamaEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type OllamaEmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

type OllamaChatMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"`
}

type OllamaChatRequest struct {
	Model    string              `json:"model"`
	Messages []OllamaChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
}

type OllamaChatResponse struct {
	Message OllamaChatMessage `json:"message"`
}

func main() {
	var dbPath string
	var apiKey string
	var ollamaURL string
	var embeddingModel string
	var visionModel string
	var useAnthropic bool
	var searchQuery string
	var webMode bool
	var port int
	var similarImage string
	var reindex bool
	var cleanup bool
	var pathFilter string
	var replace bool

	// Get default database path in home directory
	homeDir, err := os.UserHomeDir()
	if err != nil {
		homeDir = "."
	}
	defaultDB := filepath.Join(homeDir, ".lensdb.db")

	flag.StringVar(&dbPath, "db", defaultDB, "Path to SQLite database file")
	flag.StringVar(&apiKey, "api-key", "", "Anthropic API key (or set ANTHROPIC_API_KEY env var, only needed with -use-anthropic)")
	flag.StringVar(&ollamaURL, "ollama-url", "", "Ollama server URL (or set OLLAMA_URL env var, default: http://localhost:11434)")
	flag.StringVar(&embeddingModel, "embedding-model", "", "Ollama embedding model (or set OLLAMA_EMBEDDING_MODEL env var, default: nomic-embed-text)")
	flag.StringVar(&visionModel, "vision-model", "", "Ollama vision model for image descriptions (or set OLLAMA_VISION_MODEL env var, default: qwen3-vl:8b)")
	flag.BoolVar(&useAnthropic, "use-anthropic", false, "Use Anthropic Claude for image descriptions instead of Ollama vision model")
	flag.StringVar(&searchQuery, "search", "", "Search for images by description (semantic search)")
	flag.BoolVar(&webMode, "web", false, "Start web interface for searching images")
	flag.IntVar(&port, "port", 8080, "Port for web interface (default: 8080)")
	flag.StringVar(&similarImage, "similar", "", "Find images similar to the given image path")
	flag.BoolVar(&reindex, "reindex", false, "Rebuild the vector and full-text search indexes from existing descriptions")
	flag.BoolVar(&cleanup, "cleanup", false, "Remove database entries for image files that no longer exist on disk")
	flag.StringVar(&pathFilter, "path", "", "Restrict search/similar results to images within this directory path")
	flag.BoolVar(&replace, "replace", false, "Re-process and overwrite descriptions for images already in the database")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: lensdb [flags] [image-folder [db-path]]\n\n")

		fmt.Fprintf(os.Stderr, "Modes (default: process image-folder):\n")
		for _, name := range []string{"search", "similar", "web", "reindex", "cleanup"} {
			f := flag.Lookup(name)
			fmt.Fprintf(os.Stderr, "  -%s\t%s\n", f.Name, f.Usage)
		}

		fmt.Fprintf(os.Stderr, "\nProcess options:\n")
		for _, name := range []string{"replace"} {
			f := flag.Lookup(name)
			fmt.Fprintf(os.Stderr, "  -%s\t%s\n", f.Name, f.Usage)
		}

		fmt.Fprintf(os.Stderr, "\nFilters (apply to search, similar, and web modes):\n")
		for _, name := range []string{"path"} {
			f := flag.Lookup(name)
			fmt.Fprintf(os.Stderr, "  -%s\t%s\n", f.Name, f.Usage)
		}

		fmt.Fprintf(os.Stderr, "\nDatabase:\n")
		for _, name := range []string{"db"} {
			f := flag.Lookup(name)
			fmt.Fprintf(os.Stderr, "  -%s\t%s\n", f.Name, f.Usage)
		}

		fmt.Fprintf(os.Stderr, "\nOllama settings:\n")
		for _, name := range []string{"ollama-url", "vision-model", "embedding-model"} {
			f := flag.Lookup(name)
			fmt.Fprintf(os.Stderr, "  -%s\t%s\n", f.Name, f.Usage)
		}

		fmt.Fprintf(os.Stderr, "\nAnthropic settings (optional alternative for image descriptions):\n")
		for _, name := range []string{"use-anthropic", "api-key"} {
			f := flag.Lookup(name)
			fmt.Fprintf(os.Stderr, "  -%s\t%s\n", f.Name, f.Usage)
		}

		fmt.Fprintf(os.Stderr, "\nWeb server:\n")
		for _, name := range []string{"port"} {
			f := flag.Lookup(name)
			fmt.Fprintf(os.Stderr, "  -%s\t%s\n", f.Name, f.Usage)
		}
	}

	flag.Parse()

	// Normalize pathFilter to an absolute path with trailing separator so
	// strings.HasPrefix correctly scopes matches to that directory.
	if pathFilter != "" {
		if abs, err := filepath.Abs(pathFilter); err == nil {
			pathFilter = abs
		}
		if !strings.HasSuffix(pathFilter, string(filepath.Separator)) {
			pathFilter += string(filepath.Separator)
		}
	}

	// Get Ollama URL from environment if not provided
	if ollamaURL == "" {
		ollamaURL = os.Getenv("OLLAMA_URL")
		if ollamaURL == "" {
			ollamaURL = "http://localhost:11434"
		}
	}

	// Get embedding model from environment if not provided
	if embeddingModel == "" {
		embeddingModel = os.Getenv("OLLAMA_EMBEDDING_MODEL")
		if embeddingModel == "" {
			embeddingModel = "bge-m3"
		}
	}

	// Get vision model from environment if not provided
	if visionModel == "" {
		visionModel = os.Getenv("OLLAMA_VISION_MODEL")
		if visionModel == "" {
			visionModel = "qwen3-vl:8b"
		}
	}

	// Initialize sqlite-vec
	sqlite_vec.Auto()

	// Initialize database
	db, err := initDatabase(dbPath)
	if err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	defer db.Close()

	// Handle web mode
	if webMode {
		// Verify Ollama API is available before starting web server
		if err := verifyOllamaAPI(ollamaURL, embeddingModel); err != nil {
			log.Fatalf("Error: Ollama API is not available at %s\n"+
				"Please ensure Ollama is running and the model '%s' is available.\n"+
				"Details: %v", ollamaURL, embeddingModel, err)
		}

		fmt.Printf("Starting web server on http://localhost:%d\n", port)
		startWebServer(db, ollamaURL, embeddingModel, port, pathFilter)
		return
	}

	// Handle search mode
	if searchQuery != "" {
		// Verify Ollama API is available before searching
		if err := verifyOllamaAPI(ollamaURL, embeddingModel); err != nil {
			log.Fatalf("Error: Ollama API is not available at %s\n"+
				"Please ensure Ollama is running and the model '%s' is available.\n"+
				"Details: %v", ollamaURL, embeddingModel, err)
		}

		err = searchImages(db, searchQuery, ollamaURL, embeddingModel, pathFilter)
		if err != nil {
			log.Fatalf("Failed to search images: %v", err)
		}
		return
	}

	// Handle similar image mode
	if similarImage != "" {
		err = findSimilarImages(db, similarImage, pathFilter)
		if err != nil {
			log.Fatalf("Failed to find similar images: %v", err)
		}
		return
	}

	// Handle reindex mode
	if reindex {
		if err := verifyOllamaAPI(ollamaURL, embeddingModel); err != nil {
			log.Fatalf("Error: Ollama API is not available at %s\n"+
				"Please ensure Ollama is running and the model '%s' is available.\n"+
				"Details: %v", ollamaURL, embeddingModel, err)
		}
		if err := reindexDatabase(db, ollamaURL, embeddingModel); err != nil {
			log.Fatalf("Failed to reindex database: %v", err)
		}
		return
	}

	// Handle cleanup mode
	if cleanup {
		if err := cleanupDatabase(db); err != nil {
			log.Fatalf("Failed to clean up database: %v", err)
		}
		return
	}

	// Handle process mode
	if len(flag.Args()) == 0 {
		log.Fatal("Usage: lensdb [options] <folder-path>\n" +
			"  or:  lensdb -search \"query\" [options]\n" +
			"  or:  lensdb -similar \"/path/to/image.jpg\" [options]\n" +
			"  or:  lensdb -web [options]\n" +
			"  or:  lensdb -reindex [options]\n\n" +
			"Options:\n" +
			"  -db              Path to SQLite database file\n" +
			"  -ollama-url      Ollama server URL (or set OLLAMA_URL env var)\n" +
			"  -embedding-model Ollama embedding model (or set OLLAMA_EMBEDDING_MODEL env var)\n" +
			"  -vision-model    Ollama vision model for descriptions (or set OLLAMA_VISION_MODEL env var)\n" +
			"  -use-anthropic   Use Anthropic Claude for image descriptions instead of Ollama\n" +
			"  -api-key         Anthropic API key (or set ANTHROPIC_API_KEY env var, only with -use-anthropic)\n" +
			"  -search          Search for images by description\n" +
			"  -similar         Find images similar to the specified image\n" +
			"  -web             Start web interface for searching images\n" +
			"  -port            Port for web interface (default: 8080)\n" +
			"  -reindex         Rebuild vector and full-text search indexes from existing descriptions\n\n" +
			"Environment Variables:\n" +
			"  OLLAMA_URL              Ollama server URL (default: http://localhost:11434)\n" +
			"  OLLAMA_EMBEDDING_MODEL  Embedding model name (default: nomic-embed-text)\n" +
			"  OLLAMA_VISION_MODEL     Vision model for descriptions (default: qwen3-vl:8b)\n" +
			"  ANTHROPIC_API_KEY       Anthropic API key (only needed with -use-anthropic)")
	}
	folderPath := flag.Args()[0]

	// Verify Ollama embedding API is available before processing
	if err := verifyOllamaAPI(ollamaURL, embeddingModel); err != nil {
		log.Fatalf("Error: Ollama API is not available at %s\n"+
			"Please ensure Ollama is running and the model '%s' is available.\n"+
			"Details: %v", ollamaURL, embeddingModel, err)
	}

	if useAnthropic {
		// Get API key from environment if not provided
		if apiKey == "" {
			apiKey = os.Getenv("ANTHROPIC_API_KEY")
			if apiKey == "" {
				log.Fatal("Please provide an API key using -api-key flag or ANTHROPIC_API_KEY environment variable")
			}
		}

		if err := verifyAnthropicAPI(apiKey); err != nil {
			log.Fatalf("Error: Anthropic API is not available.\n"+
				"Please check your API key and network connection.\n"+
				"Details: %v", err)
		}
		fmt.Println("API verification successful. Starting image processing with Anthropic Claude...")
	} else {
		fmt.Printf("Starting image processing with Ollama vision model '%s'...\n", visionModel)
	}

	// Process images in folder
	err = processFolder(folderPath, db, apiKey, ollamaURL, embeddingModel, visionModel, useAnthropic, replace)
	if err != nil {
		log.Fatalf("Failed to process folder: %v", err)
	}

	fmt.Println("Successfully processed all images")
}

func initDatabase(dbPath string) (*sql.DB, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, err
	}

	createTableSQL := `
	CREATE TABLE IF NOT EXISTS image_descriptions (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		filename TEXT NOT NULL,
		foldername TEXT NOT NULL,
		path TEXT NOT NULL UNIQUE,
		description TEXT NOT NULL,
		embedding BLOB,
		thumbnail BLOB,
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);
	`

	_, err = db.Exec(createTableSQL)
	if err != nil {
		return nil, err
	}

	// Migrate existing databases: add thumbnail column if missing
	_, _ = db.Exec(`ALTER TABLE image_descriptions ADD COLUMN thumbnail BLOB`)

	// Migrate vec_descriptions if it was created with a different embedding dimension.
	// Drop and recreate (data will be rebuilt by -reindex).
	var vecSchema string
	db.QueryRow("SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_descriptions'").Scan(&vecSchema)
	expectedDim := fmt.Sprintf("float[%d]", embeddingDimensions)
	if vecSchema != "" && !strings.Contains(vecSchema, expectedDim) {
		if _, err = db.Exec("DROP TABLE vec_descriptions"); err != nil {
			return nil, fmt.Errorf("failed to drop outdated vec_descriptions: %w", err)
		}
		log.Printf("vec_descriptions dimension changed to %d — run -reindex to rebuild the vector index.", embeddingDimensions)
	}

	// Create virtual table for vector search if not exists
	createVecTableSQL := fmt.Sprintf(`
	CREATE VIRTUAL TABLE IF NOT EXISTS vec_descriptions USING vec0(
		embedding float[%d]
	);
	`, embeddingDimensions)

	_, err = db.Exec(createVecTableSQL)
	if err != nil {
		return nil, err
	}

	// Create FTS5 virtual table for full-text search if not exists
	createFTSSQL := `
	CREATE VIRTUAL TABLE IF NOT EXISTS fts_descriptions USING fts5(description);
	`
	_, err = db.Exec(createFTSSQL)
	if err != nil {
		return nil, err
	}

	// Migrate existing databases: populate FTS table if it is empty
	var ftsCount int
	db.QueryRow("SELECT COUNT(*) FROM fts_descriptions").Scan(&ftsCount)
	if ftsCount == 0 {
		_, _ = db.Exec(`INSERT INTO fts_descriptions(rowid, description) SELECT id, description FROM image_descriptions`)
	}

	return db, nil
}

// imageExistsInDB checks if an image path already has an entry in the database
func imageExistsInDB(db *sql.DB, imagePath string) (bool, error) {
	absPath, err := filepath.Abs(imagePath)
	if err != nil {
		absPath = imagePath
	}

	var count int
	query := "SELECT COUNT(*) FROM image_descriptions WHERE path = ?"
	err = db.QueryRow(query, absPath).Scan(&count)
	if err != nil {
		return false, err
	}

	return count > 0, nil
}

// generateEmbedding calls Ollama API to generate an embedding for the given text
func generateEmbedding(text, ollamaURL, model string) ([]float32, error) {
	reqBody := OllamaEmbeddingRequest{
		Model:  model,
		Prompt: text,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	url := ollamaURL + "/api/embeddings"
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Ollama at %s: %w", url, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Ollama API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var ollamaResp OllamaEmbeddingResponse
	err = json.Unmarshal(body, &ollamaResp)
	if err != nil {
		return nil, err
	}

	if len(ollamaResp.Embedding) == 0 {
		return nil, fmt.Errorf("no embedding in Ollama response")
	}

	return ollamaResp.Embedding, nil
}

// verifyOllamaAPI checks if the Ollama API is available and the specified model is accessible
func verifyOllamaAPI(ollamaURL, model string) error {
	reqBody := OllamaEmbeddingRequest{
		Model:  model,
		Prompt: "test",
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal test request: %w", err)
	}

	url := ollamaURL + "/api/embeddings"
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create test request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("cannot connect to Ollama server: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("Ollama API returned status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// verifyAnthropicAPI checks if the Anthropic API is available with the given API key
func verifyAnthropicAPI(apiKey string) error {
	// Create a minimal test request
	reqBody := AnthropicRequest{
		Model:     "claude-sonnet-4-5-20250929",
		MaxTokens: 1,
		Messages: []Message{
			{
				Role: "user",
				Content: []ContentItem{
					{
						Type: "text",
						Text: "Hi",
					},
				},
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal test request: %w", err)
	}

	req, err := http.NewRequest("POST", anthropicAPIURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create test request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("cannot connect to Anthropic API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("Anthropic API returned status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

func processFolder(folderPath string, db *sql.DB, apiKey, ollamaURL, embeddingModel, visionModel string, useAnthropic, replace bool) error {
	imageExtensions := map[string]bool{
		".jpg":  true,
		".jpeg": true,
		".png":  true,
		".gif":  true,
		".webp": true,
	}

	return filepath.Walk(folderPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		ext := strings.ToLower(filepath.Ext(path))
		if !imageExtensions[ext] {
			return nil
		}

		// Check if image already exists in database
		exists, err := imageExistsInDB(db, path)
		if err != nil {
			log.Printf("Failed to check if image exists in DB %s: %v", path, err)
			return nil
		}

		if exists && !replace {
			fmt.Printf("⊘ Skipping (already in DB): %s\n", path)
			return nil
		}

		fmt.Printf("Processing: %s\n", path)

		var description string
		if useAnthropic {
			description, err = describeImage(path, apiKey)
			if err != nil {
				return fmt.Errorf("failed to describe image %s - Anthropic API error: %w", path, err)
			}
		} else {
			description, err = describeImageWithOllama(path, ollamaURL, visionModel)
			if err != nil {
				return fmt.Errorf("failed to describe image %s - Ollama API error: %w", path, err)
			}
		}

		// Generate embedding from description
		embedding, err := generateEmbedding(description, ollamaURL, embeddingModel)
		if err != nil {
			return fmt.Errorf("failed to generate embedding for %s - Ollama API error: %w", path, err)
		}

		thumbnail, err := generateThumbnail(path)
		if err != nil {
			log.Printf("Warning: failed to generate thumbnail for %s: %v", path, err)
		}

		err = storeImageDescription(db, path, description, embedding, thumbnail)
		if err != nil {
			return fmt.Errorf("failed to store description for %s: %w", path, err)
		}

		fmt.Printf("✓ Stored description for: %s\n", path)
		return nil
	})
}

// resizeImage resizes an image so its longest side is maxSize pixels
func resizeImage(img image.Image, maxSize int) image.Image {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Check if resizing is needed
	if width <= maxSize && height <= maxSize {
		return img
	}

	// Calculate new dimensions
	var newWidth, newHeight int
	if width > height {
		newWidth = maxSize
		newHeight = (height * maxSize) / width
	} else {
		newHeight = maxSize
		newWidth = (width * maxSize) / height
	}

	// Create new image and resize
	dst := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))
	draw.CatmullRom.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)
	return dst
}

// generateThumbnail creates a 200px (longest side) JPEG thumbnail for storage
func generateThumbnail(imagePath string) ([]byte, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	img = resizeImage(img, 200)

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 80}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func describeImage(imagePath string, apiKey string) (string, error) {
	// Read and decode image file
	file, err := os.Open(imagePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	img, format, err := image.Decode(file)
	if err != nil {
		return "", fmt.Errorf("failed to decode image: %w", err)
	}

	// Resize image if needed (longest side = 1000px)
	img = resizeImage(img, 1000)

	// Determine media type based on original format
	ext := strings.ToLower(filepath.Ext(imagePath))
	mediaType := "image/jpeg"
	switch ext {
	case ".png":
		mediaType = "image/png"
	case ".gif":
		mediaType = "image/gif"
	case ".webp":
		mediaType = "image/webp"
	}

	// Re-encode image to bytes
	var buf bytes.Buffer
	switch format {
	case "jpeg", "jpg":
		err = jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85})
	case "png":
		err = png.Encode(&buf, img)
	case "gif":
		// For GIF, convert to paletted image
		bounds := img.Bounds()
		palettedImg := image.NewPaletted(bounds, nil)
		draw.Draw(palettedImg, bounds, img, bounds.Min, draw.Src)
		err = gif.Encode(&buf, palettedImg, nil)
	default:
		// Default to JPEG for webp and others
		err = jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85})
		mediaType = "image/jpeg"
	}
	if err != nil {
		return "", fmt.Errorf("failed to encode resized image: %w", err)
	}

	// Encode image to base64
	base64Image := base64.StdEncoding.EncodeToString(buf.Bytes())

	// Create API request
	reqBody := AnthropicRequest{
		Model:     "claude-sonnet-4-5-20250929",
		MaxTokens: 1024,
		Messages: []Message{
			{
				Role: "user",
				Content: []ContentItem{
					{
						Type: "image",
						Source: &ImageSource{
							Type:      "base64",
							MediaType: mediaType,
							Data:      base64Image,
						},
					},
					{
						Type: "text",
						Text: "Please provide a detailed description of this image.",
					},
				},
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	// Make API request
	req, err := http.NewRequest("POST", anthropicAPIURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var apiResp AnthropicResponse
	err = json.Unmarshal(body, &apiResp)
	if err != nil {
		return "", err
	}

	if len(apiResp.Content) == 0 {
		return "", fmt.Errorf("no content in API response")
	}

	return apiResp.Content[0].Text, nil
}

// describeImageWithOllama calls the Ollama chat API with a vision model to describe an image
func describeImageWithOllama(imagePath, ollamaURL, visionModel string) (string, error) {
	// Read and decode image file
	file, err := os.Open(imagePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	img, format, err := image.Decode(file)
	if err != nil {
		return "", fmt.Errorf("failed to decode image: %w", err)
	}

	// Resize image if needed (longest side = 1000px)
	img = resizeImage(img, 1000)

	// Re-encode image to JPEG bytes (Ollama vision models accept JPEG/PNG base64)
	var buf bytes.Buffer
	switch format {
	case "png":
		err = png.Encode(&buf, img)
	default:
		err = jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85})
	}
	if err != nil {
		return "", fmt.Errorf("failed to encode resized image: %w", err)
	}

	base64Image := base64.StdEncoding.EncodeToString(buf.Bytes())

	reqBody := OllamaChatRequest{
		Model:  visionModel,
		Stream: false,
		Messages: []OllamaChatMessage{
			{
				Role:    "user",
				Content: "Please provide a detailed description of this image.",
				Images:  []string{base64Image},
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	apiURL := ollamaURL + "/api/chat"
	req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to connect to Ollama at %s: %w", apiURL, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Ollama API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var ollamaResp OllamaChatResponse
	if err := json.Unmarshal(body, &ollamaResp); err != nil {
		return "", fmt.Errorf("failed to parse Ollama response: %w", err)
	}

	if ollamaResp.Message.Content == "" {
		return "", fmt.Errorf("no content in Ollama response")
	}

	return ollamaResp.Message.Content, nil
}

func storeImageDescription(db *sql.DB, imagePath string, description string, embedding []float32, thumbnail []byte) error {
	filename := filepath.Base(imagePath)
	foldername := filepath.Base(filepath.Dir(imagePath))
	absPath, err := filepath.Abs(imagePath)
	if err != nil {
		absPath = imagePath
	}

	// Serialize embedding for storage
	embeddingBlob, err := sqlite_vec.SerializeFloat32(embedding)
	if err != nil {
		return fmt.Errorf("failed to serialize embedding: %w", err)
	}

	// Start transaction
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// If a row for this path already exists, delete its FTS entry first (INSERT OR
	// REPLACE re-assigns the autoincrement id, which would orphan the old rowid).
	var oldID int64
	if err := tx.QueryRow("SELECT id FROM image_descriptions WHERE path = ?", absPath).Scan(&oldID); err == nil {
		_, _ = tx.Exec("DELETE FROM fts_descriptions WHERE rowid = ?", oldID)
	}

	// Insert or replace in main table
	insertSQL := `
	INSERT OR REPLACE INTO image_descriptions (filename, foldername, path, description, embedding, thumbnail)
	VALUES (?, ?, ?, ?, ?, ?)
	`

	result, err := tx.Exec(insertSQL, filename, foldername, absPath, description, embeddingBlob, thumbnail)
	if err != nil {
		return err
	}

	// Get the row ID
	rowID, err := result.LastInsertId()
	if err != nil {
		return err
	}

	// Insert or replace in vector table (rowid must match)
	vecInsertSQL := `
	INSERT OR REPLACE INTO vec_descriptions (rowid, embedding)
	VALUES (?, ?)
	`

	_, err = tx.Exec(vecInsertSQL, rowID, embeddingBlob)
	if err != nil {
		return err
	}

	// Insert into FTS table (rowid must match)
	_, err = tx.Exec("INSERT INTO fts_descriptions(rowid, description) VALUES(?, ?)", rowID, description)
	if err != nil {
		return err
	}

	return tx.Commit()
}

// hybridSearch runs vector search and FTS, merges results with Reciprocal Rank
// Fusion (k=60), giving FTS matches ftsWeight× the vector weight so that exact
// word matches surface above purely contextual ones.
func deserializeFloat32(data []byte) []float32 {
	n := len(data) / 4
	out := make([]float32, n)
	for i := range out {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func dotProduct(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

func hybridSearch(db *sql.DB, queryBlob []byte, ftsQuery string, limit int, pathFilter string) ([]SearchResult, error) {
	type entry struct {
		SearchResult
		id int64
	}

	const rrfK = 60.0
	const ftsWeight = 2.0

	var vecEntries []entry

	if pathFilter != "" {
		// In-memory vector search scoped to the path: no k cap applies.
		// Fetch all images matching the path, rank by dot-product similarity.
		queryEmbedding := deserializeFloat32(queryBlob)
		rows, err := db.Query(`
			SELECT id, filename, foldername, path, description, embedding, thumbnail
			FROM image_descriptions WHERE path LIKE ?`,
			pathFilter+"%")
		if err != nil {
			return nil, fmt.Errorf("vector search failed: %w", err)
		}
		defer rows.Close()
		type candidate struct {
			entry
			score float64
		}
		var candidates []candidate
		for rows.Next() {
			var e entry
			var embeddingBlob, thumbnailBlob []byte
			if err := rows.Scan(&e.id, &e.Filename, &e.Foldername, &e.Path, &e.Description, &embeddingBlob, &thumbnailBlob); err != nil {
				return nil, err
			}
			if len(thumbnailBlob) > 0 {
				e.Thumbnail = base64.StdEncoding.EncodeToString(thumbnailBlob)
			}
			candidates = append(candidates, candidate{entry: e, score: dotProduct(queryEmbedding, deserializeFloat32(embeddingBlob))})
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })
		for _, c := range candidates {
			vecEntries = append(vecEntries, c.entry)
		}
	} else {
		// Global vector search via vec0 virtual table.
		vecRows, err := db.Query(`
			SELECT img.id, img.filename, img.foldername, img.path, img.description, vec.distance, img.thumbnail
			FROM vec_descriptions vec
			INNER JOIN image_descriptions img ON vec.rowid = img.id
			WHERE vec.embedding MATCH ? AND k = ?
			ORDER BY vec.distance`,
			queryBlob, limit*5)
		if err != nil {
			return nil, fmt.Errorf("vector search failed: %w", err)
		}
		defer vecRows.Close()
		for vecRows.Next() {
			var e entry
			var thumbnailBlob []byte
			if err := vecRows.Scan(&e.id, &e.Filename, &e.Foldername, &e.Path, &e.Description, &e.Distance, &thumbnailBlob); err != nil {
				return nil, err
			}
			if len(thumbnailBlob) > 0 {
				e.Thumbnail = base64.StdEncoding.EncodeToString(thumbnailBlob)
			}
			vecEntries = append(vecEntries, e)
		}
		if err := vecRows.Err(); err != nil {
			return nil, err
		}
	}

	// --- FTS search (best-effort; fall back to vector-only on error) ---
	type ftsEntry struct {
		id int64
	}
	var ftsEntries []ftsEntry

	var ftsRows *sql.Rows
	var ftsErr error
	if pathFilter != "" {
		ftsRows, ftsErr = db.Query(`
			SELECT img.id FROM fts_descriptions fts
			JOIN image_descriptions img ON fts.rowid = img.id
			WHERE fts.description MATCH ? AND img.path LIKE ?
			ORDER BY rank`,
			ftsQuery, pathFilter+"%")
	} else {
		ftsRows, ftsErr = db.Query(`
			SELECT img.id FROM fts_descriptions fts
			JOIN image_descriptions img ON fts.rowid = img.id
			WHERE fts.description MATCH ?
			ORDER BY rank LIMIT ?`,
			ftsQuery, limit*5)
	}
	if ftsErr == nil {
		defer ftsRows.Close()
		for ftsRows.Next() {
			var fe ftsEntry
			if err := ftsRows.Scan(&fe.id); err == nil {
				ftsEntries = append(ftsEntries, fe)
			}
		}
	}
	// ftsErr != nil just means we skip FTS contribution (vector search still runs)

	// --- Reciprocal Rank Fusion ---
	type scored struct {
		entry
		score float64
	}

	scoreMap := make(map[int64]*scored)

	for rank, e := range vecEntries {
		s := 1.0 / (rrfK + float64(rank))
		cp := e
		scoreMap[e.id] = &scored{entry: cp, score: s}
	}

	for rank, fe := range ftsEntries {
		s := ftsWeight / (rrfK + float64(rank))
		if existing, ok := scoreMap[fe.id]; ok {
			existing.score += s
		} else {
			// FTS-only hit: we need the full row data
			var e entry
			var thumbnailBlob []byte
			row := db.QueryRow(`SELECT id, filename, foldername, path, description, 0.0, thumbnail FROM image_descriptions WHERE id = ?`, fe.id)
			if err := row.Scan(&e.id, &e.Filename, &e.Foldername, &e.Path, &e.Description, &e.Distance, &thumbnailBlob); err == nil {
				if len(thumbnailBlob) > 0 {
					e.Thumbnail = base64.StdEncoding.EncodeToString(thumbnailBlob)
				}
				scoreMap[fe.id] = &scored{entry: e, score: s}
			}
		}
	}

	var combined []scored
	for _, s := range scoreMap {
		combined = append(combined, *s)
	}
	sort.Slice(combined, func(i, j int) bool {
		return combined[i].score > combined[j].score
	})

	var results []SearchResult
	for _, s := range combined {
		if len(results) >= limit {
			break
		}
		results = append(results, s.SearchResult)
	}
	return results, nil
}

// searchImages performs hybrid (vector + full-text) search for images based on a query
func searchImages(db *sql.DB, query, ollamaURL, embeddingModel, pathFilter string) error {
	fmt.Printf("Searching for: %s\n\n", query)

	// Generate embedding for the query
	queryEmbedding, err := generateEmbedding(query, ollamaURL, embeddingModel)
	if err != nil {
		return fmt.Errorf("failed to generate query embedding: %w", err)
	}

	queryBlob, err := sqlite_vec.SerializeFloat32(queryEmbedding)
	if err != nil {
		return fmt.Errorf("failed to serialize query embedding: %w", err)
	}

	results, err := hybridSearch(db, queryBlob, query, 10, pathFilter)
	if err != nil {
		return err
	}

	if len(results) == 0 {
		fmt.Println("No results found.")
	}
	for i, r := range results {
		fmt.Printf("Result %d:\n", i+1)
		fmt.Printf("  File: %s\n", r.Path)
		fmt.Printf("  Description: %s\n\n", r.Description)
	}
	if len(results) > 0 {
		fmt.Printf("Found %d matching images.\n", len(results))
	}

	return nil
}

// findSimilarImages performs similarity search based on an existing image's embedding
func findSimilarImages(db *sql.DB, imagePath, pathFilter string) error {
	// Convert to absolute path for DB lookup
	absPath, err := filepath.Abs(imagePath)
	if err != nil {
		return fmt.Errorf("invalid image path: %w", err)
	}

	// Query database for embedding
	var embeddingBlob []byte
	err = db.QueryRow("SELECT embedding FROM image_descriptions WHERE path = ?", absPath).Scan(&embeddingBlob)
	if err == sql.ErrNoRows {
		return fmt.Errorf("image not found in database: %s\nRun lensdb on this image first to add it to the database", absPath)
	}
	if err != nil {
		return fmt.Errorf("failed to retrieve embedding: %w", err)
	}

	// Execute vector search with source exclusion
	fmt.Printf("\nImages similar to: %s\n\n", filepath.Base(absPath))

	type simResult struct {
		SearchResult
		score float64
	}

	var simResults []simResult

	if pathFilter != "" {
		// In-memory similarity search within the filtered path — no k cap.
		sourceEmbedding := deserializeFloat32(embeddingBlob)
		rows, err := db.Query(`
			SELECT filename, foldername, path, description, embedding
			FROM image_descriptions WHERE path LIKE ? AND path != ?`,
			pathFilter+"%", absPath)
		if err != nil {
			return fmt.Errorf("failed to search for similar images: %w", err)
		}
		defer rows.Close()
		for rows.Next() {
			var r simResult
			var emb []byte
			if err := rows.Scan(&r.Filename, &r.Foldername, &r.Path, &r.Description, &emb); err != nil {
				log.Printf("Error scanning result: %v", err)
				continue
			}
			r.score = dotProduct(sourceEmbedding, deserializeFloat32(emb))
			simResults = append(simResults, r)
		}
		sort.Slice(simResults, func(i, j int) bool { return simResults[i].score > simResults[j].score })
	} else {
		rows, err := db.Query(`
			SELECT img.filename, img.foldername, img.path, img.description, vec.distance
			FROM vec_descriptions vec
			INNER JOIN image_descriptions img ON vec.rowid = img.id
			WHERE vec.embedding MATCH ? AND k = 11 AND img.path != ?
			ORDER BY vec.distance`,
			embeddingBlob, absPath)
		if err != nil {
			return fmt.Errorf("failed to search for similar images: %w", err)
		}
		defer rows.Close()
		for rows.Next() {
			var r simResult
			if err := rows.Scan(&r.Filename, &r.Foldername, &r.Path, &r.Description, &r.Distance); err != nil {
				log.Printf("Error scanning result: %v", err)
				continue
			}
			simResults = append(simResults, r)
		}
	}

	count := 0
	for _, result := range simResults {
		if count >= 10 {
			break
		}
		fmt.Printf("File: %s\n", result.Filename)
		fmt.Printf("Folder: %s\n", result.Foldername)
		fmt.Printf("Path: %s\n", result.Path)
		fmt.Printf("Description: %s\n", result.Description)
		fmt.Printf("Distance: %.4f\n", result.Distance)
		fmt.Println("---")
		count++
	}

	if count == 0 {
		fmt.Println("No similar images found in database.")
	} else {
		fmt.Printf("\nFound %d similar images\n", count)
	}

	return nil
}

// searchImagesForWeb performs hybrid (vector + full-text) search and returns results
func searchImagesForWeb(db *sql.DB, query, ollamaURL, embeddingModel, pathFilter string) ([]SearchResult, error) {
	queryEmbedding, err := generateEmbedding(query, ollamaURL, embeddingModel)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	queryBlob, err := sqlite_vec.SerializeFloat32(queryEmbedding)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize query embedding: %w", err)
	}

	return hybridSearch(db, queryBlob, query, 25, pathFilter)
}

// findSimilarImagesForWeb performs similarity search for web interface and returns results
func findSimilarImagesForWeb(db *sql.DB, imagePath, pathFilter string) ([]SearchResult, error) {
	// Convert to absolute path for DB lookup
	absPath, err := filepath.Abs(imagePath)
	if err != nil {
		return nil, fmt.Errorf("invalid image path: %w", err)
	}

	// Query database for embedding
	var embeddingBlob []byte
	err = db.QueryRow("SELECT embedding FROM image_descriptions WHERE path = ?", absPath).Scan(&embeddingBlob)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("image not found in database")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve embedding: %w", err)
	}

	type webSimResult struct {
		SearchResult
		score float64
	}

	var simResults []webSimResult

	if pathFilter != "" {
		// In-memory similarity search within the filtered path — no k cap.
		sourceEmbedding := deserializeFloat32(embeddingBlob)
		rows, err := db.Query(`
			SELECT filename, foldername, path, description, embedding, thumbnail
			FROM image_descriptions WHERE path LIKE ? AND path != ?`,
			pathFilter+"%", absPath)
		if err != nil {
			return nil, fmt.Errorf("failed to search for similar images: %w", err)
		}
		defer rows.Close()
		for rows.Next() {
			var r webSimResult
			var emb, thumbnailBlob []byte
			if err := rows.Scan(&r.Filename, &r.Foldername, &r.Path, &r.Description, &emb, &thumbnailBlob); err != nil {
				log.Printf("Error scanning result: %v", err)
				continue
			}
			r.score = dotProduct(sourceEmbedding, deserializeFloat32(emb))
			if len(thumbnailBlob) > 0 {
				r.Thumbnail = base64.StdEncoding.EncodeToString(thumbnailBlob)
			}
			simResults = append(simResults, r)
		}
		sort.Slice(simResults, func(i, j int) bool { return simResults[i].score > simResults[j].score })
	} else {
		rows, err := db.Query(`
			SELECT img.filename, img.foldername, img.path, img.description, vec.distance, img.thumbnail
			FROM vec_descriptions vec
			INNER JOIN image_descriptions img ON vec.rowid = img.id
			WHERE vec.embedding MATCH ? AND k = 26 AND img.path != ?
			ORDER BY vec.distance`,
			embeddingBlob, absPath)
		if err != nil {
			return nil, fmt.Errorf("failed to search for similar images: %w", err)
		}
		defer rows.Close()
		for rows.Next() {
			var r webSimResult
			var thumbnailBlob []byte
			if err := rows.Scan(&r.Filename, &r.Foldername, &r.Path, &r.Description, &r.Distance, &thumbnailBlob); err != nil {
				log.Printf("Error scanning result: %v", err)
				continue
			}
			if len(thumbnailBlob) > 0 {
				r.Thumbnail = base64.StdEncoding.EncodeToString(thumbnailBlob)
			}
			simResults = append(simResults, r)
		}
	}

	var results []SearchResult
	for _, r := range simResults {
		results = append(results, r.SearchResult)
	}
	return results, nil
}

// cleanupDatabase removes rows from image_descriptions (and the associated vec/fts rows)
// where the file no longer exists on disk.
func cleanupDatabase(db *sql.DB) error {
	rows, err := db.Query(`SELECT id, path FROM image_descriptions ORDER BY id`)
	if err != nil {
		return fmt.Errorf("querying paths: %w", err)
	}
	type entry struct {
		id   int64
		path string
	}
	var entries []entry
	for rows.Next() {
		var e entry
		if err := rows.Scan(&e.id, &e.path); err != nil {
			rows.Close()
			return fmt.Errorf("scanning row: %w", err)
		}
		entries = append(entries, e)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterating rows: %w", err)
	}

	total := len(entries)
	if total == 0 {
		fmt.Println("Database is empty – nothing to clean up.")
		return nil
	}
	fmt.Printf("Checking %d entries...\n", total)

	removed := 0
	for i, e := range entries {
		fmt.Printf("\r[%d/%d] checking...", i+1, total)
		if _, err := os.Stat(e.path); err == nil {
			continue
		}
		if _, err := db.Exec(`DELETE FROM image_descriptions WHERE id = ?`, e.id); err != nil {
			fmt.Println()
			log.Printf("Failed to delete row %d (%s): %v", e.id, e.path, err)
			continue
		}
		db.Exec(`DELETE FROM vec_descriptions WHERE rowid = ?`, e.id)
		db.Exec(`DELETE FROM fts_descriptions WHERE rowid = ?`, e.id)
		fmt.Printf("\r✗ Removed: %s\n", e.path)
		removed++
	}
	fmt.Printf("\rCleanup complete: removed %d of %d entries.\n", removed, total)
	return nil
}

// reindexDatabase regenerates embeddings for every row and rebuilds vec_descriptions
// and fts_descriptions from scratch. Each row is committed individually so a partial
// run still leaves the database in a consistent state.
func reindexDatabase(db *sql.DB, ollamaURL, embeddingModel string) error {
	// Count rows for progress reporting.
	var total int
	if err := db.QueryRow("SELECT COUNT(*) FROM image_descriptions").Scan(&total); err != nil {
		return fmt.Errorf("failed to count rows: %w", err)
	}
	if total == 0 {
		fmt.Println("Database is empty – nothing to reindex.")
		return nil
	}
	fmt.Printf("Reindexing %d entries...\n", total)

	// Wipe the derived indexes; we rebuild them row by row below.
	if _, err := db.Exec("DELETE FROM vec_descriptions"); err != nil {
		return fmt.Errorf("failed to clear vec_descriptions: %w", err)
	}

	// fts5 may not be compiled into the current binary; treat it as best-effort.
	ftsAvailable := true
	if _, err := db.Exec("DELETE FROM fts_descriptions"); err != nil {
		log.Printf("Warning: could not clear fts_descriptions (%v); full-text search index will not be rebuilt.", err)
		ftsAvailable = false
	}

	// Load all rows into memory before writing so the read cursor doesn't
	// block the write transactions (SQLite only allows one writer at a time).
	type entry struct {
		id          int64
		description string
	}
	rows, err := db.Query("SELECT id, description FROM image_descriptions ORDER BY id")
	if err != nil {
		return fmt.Errorf("failed to query descriptions: %w", err)
	}
	var entries []entry
	for rows.Next() {
		var e entry
		if err := rows.Scan(&e.id, &e.description); err != nil {
			rows.Close()
			return fmt.Errorf("failed to scan row: %w", err)
		}
		entries = append(entries, e)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return fmt.Errorf("row iteration error: %w", err)
	}

	done := 0
	errors := 0
	for _, e := range entries {
		embedding, err := generateEmbedding(e.description, ollamaURL, embeddingModel)
		if err != nil {
			log.Printf("Error generating embedding for id %d: %v", e.id, err)
			errors++
			continue
		}

		embeddingBlob, err := sqlite_vec.SerializeFloat32(embedding)
		if err != nil {
			log.Printf("Error serializing embedding for id %d: %v", e.id, err)
			errors++
			continue
		}

		tx, err := db.Begin()
		if err != nil {
			log.Printf("Error starting transaction for id %d: %v", e.id, err)
			errors++
			continue
		}

		// Update stored embedding in the main table.
		if _, err := tx.Exec("UPDATE image_descriptions SET embedding = ? WHERE id = ?", embeddingBlob, e.id); err != nil {
			tx.Rollback()
			log.Printf("Error updating embedding for id %d: %v", e.id, err)
			errors++
			continue
		}

		// Rebuild vec entry.
		if _, err := tx.Exec("INSERT OR REPLACE INTO vec_descriptions (rowid, embedding) VALUES (?, ?)", e.id, embeddingBlob); err != nil {
			tx.Rollback()
			log.Printf("Error updating vec_descriptions for id %d: %v", e.id, err)
			errors++
			continue
		}

		// Rebuild FTS entry (best-effort).
		if ftsAvailable {
			if _, err := tx.Exec("INSERT INTO fts_descriptions (rowid, description) VALUES (?, ?)", e.id, e.description); err != nil {
				log.Printf("Warning: failed to update fts_descriptions for id %d: %v", e.id, err)
			}
		}

		if err := tx.Commit(); err != nil {
			log.Printf("Error committing transaction for id %d: %v", e.id, err)
			errors++
			continue
		}

		done++
		fmt.Printf("✓ [%d/%d] id=%d\n", done, total, e.id)
	}

	fmt.Printf("\nReindex complete: %d succeeded, %d failed.\n", done, errors)
	return nil
}

// startWebServer starts the HTTP server for web interface
func startWebServer(db *sql.DB, ollamaURL, embeddingModel string, port int, pathFilter string) {
	http.HandleFunc("/", handleIndex)
	http.HandleFunc("/search", makeSearchHandler(db, ollamaURL, embeddingModel, pathFilter))
	http.HandleFunc("/similar", makeSimilarHandler(db, ollamaURL, embeddingModel, pathFilter))
	http.HandleFunc("/image/", makeImageHandler())

	addr := fmt.Sprintf(":%d", port)
	log.Fatal(http.ListenAndServe(addr, nil))
}

// handleIndex serves the main HTML page
func handleIndex(w http.ResponseWriter, r *http.Request) {
	tmpl := `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LensDB - Image Search</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .search-box {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .search-form {
            display: flex;
            gap: 10px;
        }
        .search-input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        .search-input:focus {
            outline: none;
            border-color: #667eea;
        }
        .search-button {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .search-button:hover {
            transform: translateY(-2px);
        }
        .search-button:active {
            transform: translateY(0);
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: white;
            font-size: 1.2em;
        }
        .results {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .result-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            display: flex;
            flex-direction: row;
            align-items: flex-start;
        }
        .result-image {
            display: block;
            flex-shrink: 0;
            margin: 8px;
            border: 4px solid white;
            outline: 1px solid #ccc;
        }
        .result-content {
            padding: 15px 20px;
        }
        .result-filename {
            font-weight: 600;
            margin-bottom: 5px;
            color: #667eea;
            font-size: 1.1em;
        }
        .result-path {
            font-size: 0.85em;
            color: #888;
            margin-bottom: 10px;
            word-break: break-all;
        }
        .result-description {
            color: #555;
            line-height: 1.5;
            margin-bottom: 10px;
        }
        .no-results {
            background: white;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            color: #888;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .htmx-indicator {
            display: none;
        }
        .htmx-request .htmx-indicator {
            display: inline;
        }
        .htmx-request.search-button {
            opacity: 0.7;
        }
        .similar-button {
            margin-left: 10px;
            padding: 5px 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .similar-button:hover {
            transform: scale(1.05);
        }
        .similar-button:active {
            transform: scale(0.95);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 LensDB</h1>
            <p>Semantic Image Search</p>
        </div>

        <div class="search-box">
            <form class="search-form" hx-post="/search" hx-target="#results" hx-indicator="#loading">
                <input
                    type="text"
                    name="query"
                    class="search-input"
                    placeholder="Describe what you're looking for..."
                    required
                    autofocus
                >
                <button type="submit" class="search-button">
                    <span class="htmx-indicator">⏳</span>
                    <span>Search</span>
                </button>
            </form>
        </div>

        <div id="results">
            <div class="no-results">
                Enter a search query to find images
            </div>
        </div>
    </div>
</body>
</html>`

	w.Header().Set("Content-Type", "text/html")
	fmt.Fprint(w, tmpl)
}

// makeSearchHandler creates a search handler with database and config
func makeSearchHandler(db *sql.DB, ollamaURL, embeddingModel, pathFilter string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		query := r.FormValue("query")
		if query == "" {
			http.Error(w, "Query parameter required", http.StatusBadRequest)
			return
		}

		results, err := searchImagesForWeb(db, query, ollamaURL, embeddingModel, pathFilter)
		if err != nil {
			log.Printf("Search error: %v", err)
			fmt.Fprintf(w, `<div class="no-results">Error: %s</div>`, err.Error())
			return
		}

		if len(results) == 0 {
			fmt.Fprint(w, `<div class="no-results">No results found. Try a different query.</div>`)
			return
		}

		// Render results
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, `<div class="results">`)
		for _, result := range results {
			imgSrc := "/image/?path=" + url.QueryEscape(result.Path)
			if result.Thumbnail != "" {
				imgSrc = "data:image/jpeg;base64," + result.Thumbnail
			}
			fmt.Fprintf(w, `
			<div class="result-card">
				<img src="%s" alt="%s" class="result-image" loading="lazy">
				<div class="result-content">
					<div class="result-filename">%s</div>
					<div class="result-path">%s</div>
					<div class="result-description">%s</div>
					<div>
						<button class="similar-button" hx-post="/similar" hx-target="#results" hx-vals='{"path": "%s"}'>Find Similar</button>
					</div>
				</div>
			</div>`,
				imgSrc,
				template.HTMLEscapeString(result.Filename),
				template.HTMLEscapeString(result.Filename),
				template.HTMLEscapeString(result.Path),
				template.HTMLEscapeString(result.Description),
				template.JSEscapeString(result.Path),
			)
		}
		fmt.Fprint(w, `</div>`)
	}
}

// makeSimilarHandler creates a handler for finding similar images
func makeSimilarHandler(db *sql.DB, ollamaURL, embeddingModel, pathFilter string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		imagePath := r.FormValue("path")
		if imagePath == "" {
			http.Error(w, "Path parameter required", http.StatusBadRequest)
			return
		}

		results, err := findSimilarImagesForWeb(db, imagePath, pathFilter)
		if err != nil {
			log.Printf("Similar search error: %v", err)
			if err.Error() == "image not found in database" {
				fmt.Fprint(w, `<div class="no-results">This image is not in the database.</div>`)
			} else {
				fmt.Fprintf(w, `<div class="no-results">Error: %s</div>`, template.HTMLEscapeString(err.Error()))
			}
			return
		}

		if len(results) == 0 {
			fmt.Fprint(w, `<div class="no-results">No similar images found.</div>`)
			return
		}

		// Render results using same HTML structure as search
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, `<div class="results">`)

		for _, result := range results {
			imgSrc := "/image/?path=" + url.QueryEscape(result.Path)
			if result.Thumbnail != "" {
				imgSrc = "data:image/jpeg;base64," + result.Thumbnail
			}
			fmt.Fprintf(w, `
			<div class="result-card">
				<img src="%s" alt="%s" class="result-image" loading="lazy">
				<div class="result-content">
					<div class="result-filename">%s</div>
					<div class="result-path">%s</div>
					<div class="result-description">%s</div>
					<div>
						<button class="similar-button" hx-post="/similar" hx-target="#results" hx-vals='{"path": "%s"}'>Find Similar</button>
					</div>
				</div>
			</div>`,
				imgSrc,
				template.HTMLEscapeString(result.Filename),
				template.HTMLEscapeString(result.Filename),
				template.HTMLEscapeString(result.Path),
				template.HTMLEscapeString(result.Description),
				template.JSEscapeString(result.Path),
			)
		}

		fmt.Fprint(w, `</div>`)
	}
}

// makeImageHandler creates a handler that serves image files
func makeImageHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		imagePath := r.URL.Query().Get("path")
		if imagePath == "" {
			http.Error(w, "Path parameter required", http.StatusBadRequest)
			return
		}

		// Check if file exists
		if _, err := os.Stat(imagePath); os.IsNotExist(err) {
			http.Error(w, "Image not found", http.StatusNotFound)
			return
		}

		// Determine content type based on extension
		ext := strings.ToLower(filepath.Ext(imagePath))
		contentType := "image/jpeg"
		switch ext {
		case ".png":
			contentType = "image/png"
		case ".gif":
			contentType = "image/gif"
		case ".webp":
			contentType = "image/webp"
		}

		w.Header().Set("Content-Type", contentType)
		http.ServeFile(w, r, imagePath)
	}
}
