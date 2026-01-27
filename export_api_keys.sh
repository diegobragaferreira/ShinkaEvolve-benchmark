#!/bin/bash

# load-env.sh
set -e

# Executar o script Python e capturar os comandos export
ENV_COMMANDS=$(python3 retrieve_api_keys.py)

if [ $? -eq 0 ]; then
    eval "$ENV_COMMANDS"
    echo "Environment variables exported successfully"
else
    echo "Failed to load environment variables" >&2
    exit 1
fi