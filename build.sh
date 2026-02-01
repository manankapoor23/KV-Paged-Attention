#!/bin/bash
# Render.com build script - ensures all dependencies are installed

set -e

echo "🔧 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "✓ Dependencies installed successfully"
echo ""
echo "📦 Installed packages:"
pip list | grep -E "fastapi|uvicorn|torch|transformers"
