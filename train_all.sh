#!/bin/bash

# Train all agent architectures sequentially

AGENTS=("DeepPPOAgent" "PPOAgent")

echo "======================================"
echo "Training all agent architectures"
echo "======================================"
echo ""

for agent in "${AGENTS[@]}"; do
    echo ""
    echo "======================================"
    echo "Starting training: $agent"
    echo "======================================"
    echo ""
    
    python3 src/train.py --agent "$agent"
    
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "❌ Training failed for $agent (exit code: $exit_code)"
        echo "Stopping batch training."
        exit $exit_code
    fi
    
    echo ""
    echo "✅ Completed training: $agent"
    echo ""
done

echo ""
echo "======================================"
echo "All agents trained successfully!"
echo "======================================"
