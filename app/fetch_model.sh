#!/usr/bin/env bash
set -e

# Fetch model from W&B only when variables are available
if [[ -n "$WANDB_MODEL_URL" && -n "$WANDB_API_KEY" ]]; then
    if [[ ${#WANDB_API_KEY} -eq 40 ]]; then
        echo "Logging into Weights & Biases..."
        wandb login --relogin "$WANDB_API_KEY"
        echo "Downloading model artifact $WANDB_MODEL_URL"
        wandb artifact get "$WANDB_MODEL_URL":latest -p /app/model
    else
        echo "WANDB_API_KEY appears invalid; skipping artifact download" >&2
    fi
else
    echo "WANDB environment variables not set; skipping artifact download" >&2
fi

exec "$@"
