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
    optimize)
        echo "==> Executando experimentos do problema 40..."
        python -m src.optimization.experiment
        ;;
    validate-optimization)
        echo "==> Validando avaliador, restrições, decodificação e reparo..."
        python -m src.optimization.experiment --validate-only
        ;;
    report)
        echo "==> Gerando reports/relatorio.pdf a partir de results/..."
        python reports/build_report.py
        ;;
    test)
        echo "==> Executando testes automatizados do problema 40..."
        python -m unittest discover -s tests -v
        ;;
    *)
        echo "Uso: ./run.sh {preprocess|evaluate|all|optimize|validate-optimization|report|test}"
        exit 1
        ;;
esac
