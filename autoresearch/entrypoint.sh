#!/usr/bin/env bash
set -euo pipefail

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set."
    echo "Pass it at runtime, for example:"
    echo "  docker run --gpus all -e ANTHROPIC_API_KEY=sk-ant-... deepsculpt-autoresearch"
    exit 1
fi

export ANTHROPIC_API_KEY
export AUTORESEARCH_MODE="${AUTORESEARCH_MODE:-gan}"
export AUTORESEARCH_DATA_OUT="${AUTORESEARCH_DATA_OUT:-/workspace/autoresearch_data}"
export AUTORESEARCH_RUN_OUT="${AUTORESEARCH_RUN_OUT:-/workspace/autoresearch_runs}"

mkdir -p "$AUTORESEARCH_DATA_OUT" "$AUTORESEARCH_RUN_OUT" /workspace/autoresearch
touch /workspace/autoresearch/claude.log

cat > /tmp/deepsculpt_autoresearch_prompt.txt <<PROMPT_EOF
Please read autoresearch/program.md and start the setup first.

Environment for this run:
- AUTORESEARCH_MODE=${AUTORESEARCH_MODE}
- AUTORESEARCH_DATA_OUT=${AUTORESEARCH_DATA_OUT}
- AUTORESEARCH_RUN_OUT=${AUTORESEARCH_RUN_OUT}

Create the branch, initialize autoresearch/results.tsv if needed, run a baseline, then begin the autonomous experiment loop.
PROMPT_EOF

echo "=== Starting DeepSculpt autoresearch ==="
echo "Mode: ${AUTORESEARCH_MODE}"
echo "Data output: ${AUTORESEARCH_DATA_OUT}"
echo "Run output: ${AUTORESEARCH_RUN_OUT}"

exec claude --dangerously-skip-permissions \
    -p "$(cat /tmp/deepsculpt_autoresearch_prompt.txt)" \
    --verbose 2>&1 | tee /workspace/autoresearch/claude.log
