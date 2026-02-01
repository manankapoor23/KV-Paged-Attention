#!/bin/bash
# Start the KV-Paged Visualizer (both backend and frontend)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Starting KV-Paged Visualizer..."
echo ""

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Virtual environment not found!"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Install backend dependencies
echo "Installing backend dependencies..."
$SCRIPT_DIR/venv/bin/pip install fastapi uvicorn pydantic -q

# Kill any existing processes on our ports
pkill -f "http.server 9000" 2>/dev/null || true
pkill -f "uvicorn.*8000" 2>/dev/null || true
sleep 1

# Start backend
echo "Starting backend server..."
echo "Backend: http://localhost:8000"
echo ""

cd "$SCRIPT_DIR"
$SCRIPT_DIR/venv/bin/python -m uvicorn backend.main:app --reload --port 8000 > /tmp/kvpaged-backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend HTTP server (must run from frontend directory)
echo "Starting frontend server..."
echo "Frontend: http://localhost:9000"
echo ""

cd "$SCRIPT_DIR/frontend"
/usr/bin/python3 -m http.server 9000 --bind 127.0.0.1 > /tmp/kvpaged-frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 1

# Try to open in browser
if command -v open &> /dev/null; then
    open "http://localhost:9000"
elif command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:9000"
fi

echo "Visualizer is running!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:9000"
echo ""
echo "Press Ctrl+C to stop"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

wait
