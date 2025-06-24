#!/usr/bin/env bash
set -e

# faz login no W&B com o token em $WANDB_API_KEY
wandb login --relogin "$WANDB_API_KEY"

# baixa o artifact mais recente do seu modelo
# atenção ao path exato do artifact no seu projeto W&B!
wandb artifact get "$WANDB_MODEL_URL":latest -p /app/model

# ao final, /app/model/model.keras estará disponível
# executa o comando padrão
exec "$@"
