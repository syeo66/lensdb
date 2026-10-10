package main

import (
	"database/sql"
	"encoding/json"
	"flag"
	"html"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
)

func TestMain(m *testing.M) {
	sqlite_vec.Auto()
	os.Exit(m.Run())
}

func newTestDB(t *testing.T) (*sql.DB, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.db")
	db, err := initDatabase(path)
	if err != nil {
		t.Fatalf("initDatabase: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db, path
}

// testEmbedding returns a unit vector pointing along dimension i.
func testEmbedding(i int) []float32 {
	v := make([]float32, embeddingDimensions)
	v[i] = 1
	return v
}

func countRows(t *testing.T, db *sql.DB, query string, args ...any) int {
	t.Helper()
	var n int
	if err := db.QueryRow(query, args...).Scan(&n); err != nil {
		t.Fatalf("%s: %v", query, err)
	}
	return n
}

// assertIndexesConsistent checks that the vec and fts tables have exactly one
// row per image and none for ids that no longer exist.
func assertIndexesConsistent(t *testing.T, db *sql.DB) {
	t.Helper()
	images := countRows(t, db, `SELECT COUNT(*) FROM image_descriptions`)
	if n := countRows(t, db, `SELECT COUNT(*) FROM vec_descriptions`); n != images {
		t.Errorf("vec_descriptions has %d rows, want %d", n, images)
	}
	if n := countRows(t, db, `SELECT COUNT(*) FROM fts_descriptions`); n != images {
		t.Errorf("fts_descriptions has %d rows, want %d", n, images)
	}
	if n := countRows(t, db, `SELECT COUNT(*) FROM vec_descriptions WHERE rowid NOT IN (SELECT id FROM image_descriptions)`); n != 0 {
		t.Errorf("%d orphaned vec rows", n)
	}
}

func TestParseArgs(t *testing.T) {
	tests := []struct {
		args       []string
		db         string
		positional []string
	}{
		{[]string{"-db", "x.db", "pics"}, "x.db", []string{"pics"}},
		{[]string{"pics", "-db", "x.db"}, "x.db", []string{"pics"}},
		{[]string{"pics"}, "default.db", []string{"pics"}},
		{nil, "default.db", nil},
	}
	for _, tt := range tests {
		fs := flag.NewFlagSet("test", flag.ContinueOnError)
		db := fs.String("db", "default.db", "")
		got := parseArgs(fs, tt.args)
		if *db != tt.db {
			t.Errorf("parseArgs(%q): db = %q, want %q", tt.args, *db, tt.db)
		}
		if strings.Join(got, ",") != strings.Join(tt.positional, ",") {
			t.Errorf("parseArgs(%q) = %q, want %q", tt.args, got, tt.positional)
		}
	}
}

func TestStoreReplaceKeepsID(t *testing.T) {
	db, _ := newTestDB(t)
	path := filepath.Join(t.TempDir(), "a.jpg")

	if err := storeImageDescription(db, path, "first", testEmbedding(0), nil); err != nil {
		t.Fatal(err)
	}
	var id1 int64
	db.QueryRow(`SELECT id FROM image_descriptions WHERE path = ?`, path).Scan(&id1)

	if err := storeImageDescription(db, path, "second", testEmbedding(1), nil); err != nil {
		t.Fatal(err)
	}
	var id2 int64
	var desc string
	db.QueryRow(`SELECT id, description FROM image_descriptions WHERE path = ?`, path).Scan(&id2, &desc)

	if id1 != id2 {
		t.Errorf("replace changed id from %d to %d", id1, id2)
	}
	if desc != "second" {
		t.Errorf("description = %q, want %q", desc, "second")
	}
	if n := countRows(t, db, `SELECT COUNT(*) FROM fts_descriptions WHERE fts_descriptions MATCH 'first'`); n != 0 {
		t.Errorf("old description still in FTS index")
	}
	assertIndexesConsistent(t, db)
}

func TestInitDatabaseRemovesOrphans(t *testing.T) {
	db, path := newTestDB(t)
	if err := storeImageDescription(db, filepath.Join(t.TempDir(), "a.jpg"), "kept", testEmbedding(0), nil); err != nil {
		t.Fatal(err)
	}
	blob, _ := sqlite_vec.SerializeFloat32(testEmbedding(1))
	if _, err := db.Exec(`INSERT INTO vec_descriptions (rowid, embedding) VALUES (999, ?)`, blob); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO fts_descriptions (rowid, description) VALUES (999, 'orphan')`); err != nil {
		t.Fatal(err)
	}
	db.Close()

	db, err := initDatabase(path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	assertIndexesConsistent(t, db)
	if n := countRows(t, db, `SELECT COUNT(*) FROM image_descriptions`); n != 1 {
		t.Errorf("image_descriptions has %d rows, want 1", n)
	}
}

func TestHybridSearchPathFilter(t *testing.T) {
	db, _ := newTestDB(t)
	dir := t.TempDir()
	inside := filepath.Join(dir, "my_trip", "a.jpg")
	// "_" is a LIKE wildcard; a sibling folder must not match the filter.
	outside := filepath.Join(dir, "myXtrip", "b.jpg")
	for i, p := range []string{inside, outside} {
		if err := storeImageDescription(db, p, "a sunset", testEmbedding(i), nil); err != nil {
			t.Fatal(err)
		}
	}
	query, _ := sqlite_vec.SerializeFloat32(testEmbedding(0))

	all, err := hybridSearch(db, query, "sunset", 10, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 2 {
		t.Fatalf("unfiltered search returned %d results, want 2", len(all))
	}
	if all[0].Path != inside {
		t.Errorf("best match = %s, want %s", all[0].Path, inside)
	}

	filtered, err := hybridSearch(db, query, "sunset", 10, filepath.Join(dir, "my_trip")+string(filepath.Separator))
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range filtered {
		if r.Path != inside {
			t.Errorf("path filter returned %s", r.Path)
		}
	}
}

func TestRenderResultsEscaping(t *testing.T) {
	path := `/photos/it's here/bob's "x" <pic>.png`
	rec := httptest.NewRecorder()
	renderResults(rec, []SearchResult{{
		Filename:    filepath.Base(path),
		Path:        path,
		Description: "<script>alert(1)</script>",
		Thumbnail:   "AAAA",
	}})
	body := rec.Body.String()

	if strings.Contains(body, "<script>") || strings.Contains(body, "<pic>") {
		t.Errorf("unescaped content in output:\n%s", body)
	}
	if !strings.Contains(body, `src="data:image/jpeg;base64,AAAA"`) {
		t.Errorf("thumbnail data URL missing:\n%s", body)
	}

	m := regexp.MustCompile(`hx-vals="([^"]*)"`).FindStringSubmatch(body)
	if m == nil {
		t.Fatalf("no hx-vals attribute:\n%s", body)
	}
	var vals map[string]string
	if err := json.Unmarshal([]byte(html.UnescapeString(m[1])), &vals); err != nil {
		t.Fatalf("hx-vals is not valid JSON: %v", err)
	}
	if vals["path"] != path {
		t.Errorf("hx-vals path = %q, want %q", vals["path"], path)
	}
}

func TestImageHandlerOnlyServesIndexed(t *testing.T) {
	db, _ := newTestDB(t)
	dir := t.TempDir()
	indexed := filepath.Join(dir, "indexed.png")
	other := filepath.Join(dir, "other.png")
	for _, p := range []string{indexed, other} {
		os.WriteFile(p, []byte("data"), 0o644)
	}
	if err := storeImageDescription(db, indexed, "x", testEmbedding(0), nil); err != nil {
		t.Fatal(err)
	}

	handler := makeImageHandler(db)
	for path, want := range map[string]int{
		indexed:       http.StatusOK,
		other:         http.StatusNotFound,
		"/etc/passwd": http.StatusNotFound,
	} {
		rec := httptest.NewRecorder()
		handler(rec, httptest.NewRequest("GET", "/image/?path="+url.QueryEscape(path), nil))
		if rec.Code != want {
			t.Errorf("GET %s: status %d, want %d", path, rec.Code, want)
		}
	}
}

func TestCleanupDatabase(t *testing.T) {
	db, _ := newTestDB(t)
	dir := t.TempDir()
	present := filepath.Join(dir, "present.jpg")
	deleted := filepath.Join(dir, "deleted.jpg")
	unmounted := filepath.Join(dir, "Volume", "sub", "c.jpg")
	os.WriteFile(present, []byte("data"), 0o644)
	for i, p := range []string{present, deleted, unmounted} {
		if err := storeImageDescription(db, p, "x", testEmbedding(i), nil); err != nil {
			t.Fatal(err)
		}
	}
	exists := func(p string) bool {
		return countRows(t, db, `SELECT COUNT(*) FROM image_descriptions WHERE path = ?`, p) == 1
	}

	// Declining keeps entries whose folder is missing.
	if err := cleanupDatabase(db, strings.NewReader("\n")); err != nil {
		t.Fatal(err)
	}
	if !exists(present) || exists(deleted) || !exists(unmounted) {
		t.Errorf("after declining: present=%v deleted=%v unmounted=%v, want true false true",
			exists(present), exists(deleted), exists(unmounted))
	}
	assertIndexesConsistent(t, db)

	if err := cleanupDatabase(db, strings.NewReader("y\n")); err != nil {
		t.Fatal(err)
	}
	if !exists(present) || exists(unmounted) {
		t.Errorf("after confirming: present=%v unmounted=%v, want true false", exists(present), exists(unmounted))
	}
	assertIndexesConsistent(t, db)
}

func TestTopmostMissingDir(t *testing.T) {
	dir := t.TempDir()
	want := filepath.Join(dir, "Volume")
	if got := topmostMissingDir(filepath.Join(want, "a", "b", "c.jpg")); got != want {
		t.Errorf("topmostMissingDir = %s, want %s", got, want)
	}
}
