#!/bin/sh
set -e

# Check if folder argument is provided
if [ -z "$1" ]; then
    echo "Usage: ./run.sh <folder> [db-path]"
    echo "  folder    Path to folder containing images (required)"
    echo "  db-path   Path to SQLite database file (optional, default: ~/.lensdb.db)"
    exit 1
fi

FOLDER="$1"
DB_PATH="${2:-$HOME/.lensdb.db}"

tmpFile=$(mktemp)
go build -o "$tmpFile" *.go
dotenvx run -- $tmpFile "$FOLDER" -db "$DB_PATH"

