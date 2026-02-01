#!/bin/bash
# Start multiple MLX model servers on different ports
# Usage: ./start_mlx_servers.sh [num_instances] [model]

NUM_INSTANCES=${1:-3}
MODEL=${2:-"mlx-community/Qwen2.5-7B-Instruct-4bit"}  # MLX-optimized model

echo "=============================================="
echo "Starting $NUM_INSTANCES MLX model servers"
echo "Model: $MODEL"
echo "=============================================="

# Kill any existing servers
pkill -f "mlx_lm.server" 2>/dev/null
sleep 1

# Start servers on ports 8080, 8081, 8082, etc.
BASE_PORT=8080
PIDS=()

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    echo "Starting server $i on port $PORT..."

    python3 -m mlx_lm.server \
        --model "$MODEL" \
        --port $PORT \
        --host 0.0.0.0 \
        > /tmp/mlx_server_$PORT.log 2>&1 &

    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"
    sleep 2  # Give each server time to load
done

echo ""
echo "=============================================="
echo "Servers started on ports: $BASE_PORT - $((BASE_PORT + NUM_INSTANCES - 1))"
echo "=============================================="
echo ""
echo "Test with:"
echo "  curl http://localhost:8080/v1/models"
echo ""
echo "Logs at: /tmp/mlx_server_*.log"
echo ""
echo "To stop all: pkill -f 'mlx_lm.server'"

# Wait for all
wait
