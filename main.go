package main

import (
	"bytes"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	_ "github.com/mattn/go-sqlite3"
	"golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
)

const anthropicAPIURL = "https://api.anthropic.com/v1/messages"

type ImageDescription struct {
	Filename    string
	Foldername  string
	Path        string
	Description string
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

func main() {
	var dbPath string
	var apiKey string
	var ollamaURL string
	var embeddingModel string
	var searchQuery string

	// Get default database path in home directory
	homeDir, err := os.UserHomeDir()
	if err != nil {
		homeDir = "."
	}
	defaultDB := filepath.Join(homeDir, ".lensdb.db")

	flag.StringVar(&dbPath, "db", defaultDB, "Path to SQLite database file")
	flag.StringVar(&apiKey, "api-key", "", "Anthropic API key (or set ANTHROPIC_API_KEY env var)")
	flag.StringVar(&ollamaURL, "ollama-url", "", "Ollama server URL (or set OLLAMA_URL env var, default: http://localhost:11434)")
	flag.StringVar(&embeddingModel, "embedding-model", "", "Ollama embedding model (or set OLLAMA_EMBEDDING_MODEL env var, default: nomic-embed-text)")
	flag.StringVar(&searchQuery, "search", "", "Search for images by description (semantic search)")
	flag.Parse()

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
			embeddingModel = "nomic-embed-text"
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

	// Handle search mode
	if searchQuery != "" {
		// Verify Ollama API is available before searching
		if err := verifyOllamaAPI(ollamaURL, embeddingModel); err != nil {
			log.Fatalf("Error: Ollama API is not available at %s\n"+
				"Please ensure Ollama is running and the model '%s' is available.\n"+
				"Details: %v", ollamaURL, embeddingModel, err)
		}

		err = searchImages(db, searchQuery, ollamaURL, embeddingModel)
		if err != nil {
			log.Fatalf("Failed to search images: %v", err)
		}
		return
	}

	// Handle process mode
	if len(flag.Args()) == 0 {
		log.Fatal("Usage: lensdb <folder-path> [options]\n" +
			"  or:  lensdb -search \"query\" [options]\n\n" +
			"Options:\n" +
			"  -db              Path to SQLite database file\n" +
			"  -api-key         Anthropic API key (or set ANTHROPIC_API_KEY env var)\n" +
			"  -ollama-url      Ollama server URL (or set OLLAMA_URL env var)\n" +
			"  -embedding-model Ollama embedding model (or set OLLAMA_EMBEDDING_MODEL env var)\n" +
			"  -search          Search for images by description\n\n" +
			"Environment Variables:\n" +
			"  ANTHROPIC_API_KEY       Anthropic API key for image descriptions\n" +
			"  OLLAMA_URL              Ollama server URL (default: http://localhost:11434)\n" +
			"  OLLAMA_EMBEDDING_MODEL  Embedding model name (default: nomic-embed-text)")
	}
	folderPath := flag.Args()[0]

	// Get API key from environment if not provided
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			log.Fatal("Please provide an API key using -api-key flag or ANTHROPIC_API_KEY environment variable")
		}
	}

	// Verify both APIs are available before processing
	if err := verifyOllamaAPI(ollamaURL, embeddingModel); err != nil {
		log.Fatalf("Error: Ollama API is not available at %s\n"+
			"Please ensure Ollama is running and the model '%s' is available.\n"+
			"Details: %v", ollamaURL, embeddingModel, err)
	}

	if err := verifyAnthropicAPI(apiKey); err != nil {
		log.Fatalf("Error: Anthropic API is not available.\n"+
			"Please check your API key and network connection.\n"+
			"Details: %v", err)
	}

	fmt.Println("API verification successful. Starting image processing...")

	// Process images in folder
	err = processFolder(folderPath, db, apiKey, ollamaURL, embeddingModel)
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
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);
	`

	_, err = db.Exec(createTableSQL)
	if err != nil {
		return nil, err
	}

	// Create virtual table for vector search if not exists
	createVecTableSQL := `
	CREATE VIRTUAL TABLE IF NOT EXISTS vec_descriptions USING vec0(
		embedding float[768]
	);
	`

	_, err = db.Exec(createVecTableSQL)
	if err != nil {
		return nil, err
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
		Model:     "claude-3-5-sonnet-20241022",
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

func processFolder(folderPath string, db *sql.DB, apiKey, ollamaURL, embeddingModel string) error {
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

		if exists {
			fmt.Printf("⊘ Skipping (already in DB): %s\n", path)
			return nil
		}

		fmt.Printf("Processing: %s\n", path)

		description, err := describeImage(path, apiKey)
		if err != nil {
			return fmt.Errorf("failed to describe image %s - Anthropic API error: %w", path, err)
		}

		// Generate embedding from description
		embedding, err := generateEmbedding(description, ollamaURL, embeddingModel)
		if err != nil {
			return fmt.Errorf("failed to generate embedding for %s - Ollama API error: %w", path, err)
		}

		err = storeImageDescription(db, path, description, embedding)
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
		Model:     "claude-3-5-sonnet-20241022",
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

func storeImageDescription(db *sql.DB, imagePath string, description string, embedding []float32) error {
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

	// Insert or replace in main table
	insertSQL := `
	INSERT OR REPLACE INTO image_descriptions (filename, foldername, path, description, embedding)
	VALUES (?, ?, ?, ?, ?)
	`

	result, err := tx.Exec(insertSQL, filename, foldername, absPath, description, embeddingBlob)
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

	return tx.Commit()
}

// searchImages performs semantic search for images based on a query
func searchImages(db *sql.DB, query, ollamaURL, embeddingModel string) error {
	fmt.Printf("Searching for: %s\n\n", query)

	// Generate embedding for the query
	queryEmbedding, err := generateEmbedding(query, ollamaURL, embeddingModel)
	if err != nil {
		return fmt.Errorf("failed to generate query embedding: %w", err)
	}

	// Serialize query embedding
	queryBlob, err := sqlite_vec.SerializeFloat32(queryEmbedding)
	if err != nil {
		return fmt.Errorf("failed to serialize query embedding: %w", err)
	}

	// Search for similar vectors
	// sqlite-vec requires k parameter in WHERE clause for knn queries
	searchSQL := `
	SELECT
		img.filename,
		img.foldername,
		img.path,
		img.description,
		vec.distance
	FROM vec_descriptions vec
	INNER JOIN image_descriptions img ON vec.rowid = img.id
	WHERE vec.embedding MATCH ? AND k = ?
	ORDER BY vec.distance
	`

	rows, err := db.Query(searchSQL, queryBlob, 10)
	if err != nil {
		return fmt.Errorf("search query failed: %w", err)
	}
	defer rows.Close()

	results := 0
	for rows.Next() {
		var filename, foldername, path, description string
		var distance float64

		err := rows.Scan(&filename, &foldername, &path, &description, &distance)
		if err != nil {
			return err
		}

		results++
		fmt.Printf("Result %d (similarity: %.4f):\n", results, 1.0-distance)
		fmt.Printf("  File: %s\n", path)
		fmt.Printf("  Description: %s\n\n", description)
	}

	if results == 0 {
		fmt.Println("No results found.")
	} else {
		fmt.Printf("Found %d matching images.\n", results)
	}

	return rows.Err()
}
