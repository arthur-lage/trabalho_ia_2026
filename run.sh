#!/usr/bin/env bash

set -e

COMMAND=$1

case "$COMMAND" in
    preprocess)
        echo "==> Executando pré-processamento..."
        python src/supervised/preprocess.py
        ;;
    evaluate)
        echo "==> Avaliando modelos e gerando resultados..."
        python src/supervised/evaluate.py
        ;;
    all)
        echo "==> Executando pipeline..."
        python src/supervised/preprocess.py
        python src/supervised/evaluate.py
        ;;
    *)
        echo "Uso: ./run.sh {preprocess|train|evaluate|all}"
        exit 1
        ;;
esac