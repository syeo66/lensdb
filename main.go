package main

import (
	"bytes"
	"database/sql"
	"encoding/base64"
	"encoding/json"
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

type SearchResult struct {
	Filename    string
	Foldername  string
	Path        string
	Description string
	Distance    float64
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
	var webMode bool
	var port int

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
	flag.BoolVar(&webMode, "web", false, "Start web interface for searching images")
	flag.IntVar(&port, "port", 8080, "Port for web interface (default: 8080)")
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

	// Handle web mode
	if webMode {
		// Verify Ollama API is available before starting web server
		if err := verifyOllamaAPI(ollamaURL, embeddingModel); err != nil {
			log.Fatalf("Error: Ollama API is not available at %s\n"+
				"Please ensure Ollama is running and the model '%s' is available.\n"+
				"Details: %v", ollamaURL, embeddingModel, err)
		}

		fmt.Printf("Starting web server on http://localhost:%d\n", port)
		startWebServer(db, ollamaURL, embeddingModel, port)
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

		err = searchImages(db, searchQuery, ollamaURL, embeddingModel)
		if err != nil {
			log.Fatalf("Failed to search images: %v", err)
		}
		return
	}

	// Handle process mode
	if len(flag.Args()) == 0 {
		log.Fatal("Usage: lensdb <folder-path> [options]\n" +
			"  or:  lensdb -search \"query\" [options]\n" +
			"  or:  lensdb -web [options]\n\n" +
			"Options:\n" +
			"  -db              Path to SQLite database file\n" +
			"  -api-key         Anthropic API key (or set ANTHROPIC_API_KEY env var)\n" +
			"  -ollama-url      Ollama server URL (or set OLLAMA_URL env var)\n" +
			"  -embedding-model Ollama embedding model (or set OLLAMA_EMBEDDING_MODEL env var)\n" +
			"  -search          Search for images by description\n" +
			"  -web             Start web interface for searching images\n" +
			"  -port            Port for web interface (default: 8080)\n\n" +
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

// searchImagesForWeb performs semantic search and returns results instead of printing them
func searchImagesForWeb(db *sql.DB, query, ollamaURL, embeddingModel string) ([]SearchResult, error) {
	// Generate embedding for the query
	queryEmbedding, err := generateEmbedding(query, ollamaURL, embeddingModel)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	// Serialize query embedding
	queryBlob, err := sqlite_vec.SerializeFloat32(queryEmbedding)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize query embedding: %w", err)
	}

	// Search for similar vectors
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

	rows, err := db.Query(searchSQL, queryBlob, 25)
	if err != nil {
		return nil, fmt.Errorf("search query failed: %w", err)
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var filename, foldername, path, description string
		var distance float64

		err := rows.Scan(&filename, &foldername, &path, &description, &distance)
		if err != nil {
			return nil, err
		}

		results = append(results, SearchResult{
			Filename:    filename,
			Foldername:  foldername,
			Path:        path,
			Description: description,
			Distance:    distance,
		})
	}

	return results, rows.Err()
}

// startWebServer starts the HTTP server for web interface
func startWebServer(db *sql.DB, ollamaURL, embeddingModel string, port int) {
	http.HandleFunc("/", handleIndex)
	http.HandleFunc("/search", makeSearchHandler(db, ollamaURL, embeddingModel))
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
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        .result-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }
        .result-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
            background: #f0f0f0;
        }
        .result-content {
            padding: 20px;
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
        .result-distance {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
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
func makeSearchHandler(db *sql.DB, ollamaURL, embeddingModel string) http.HandlerFunc {
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

		results, err := searchImagesForWeb(db, query, ollamaURL, embeddingModel)
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
			encodedPath := url.QueryEscape(result.Path)
			fmt.Fprintf(w, `
			<div class="result-card">
				<img src="/image/?path=%s" alt="%s" class="result-image" loading="lazy">
				<div class="result-content">
					<div class="result-filename">%s</div>
					<div class="result-path">%s</div>
					<div class="result-description">%s</div>
					<span class="result-distance">distance: %.4f</span>
				</div>
			</div>`,
				encodedPath,
				template.HTMLEscapeString(result.Filename),
				template.HTMLEscapeString(result.Filename),
				template.HTMLEscapeString(result.Path),
				template.HTMLEscapeString(result.Description),
				result.Distance,
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
