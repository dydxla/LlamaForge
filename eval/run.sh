#!/bin/bash

# 기본값 설정
MODEL_PATH=""
TOKENIZER_PATH=""
BENCHMARKS=""

# 옵션 파싱
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --tokenizer_path) TOKENIZER_PATH="$2"; shift ;;
        --benchmarks) BENCHMARKS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# 실행 로그
echo "Starting evaluation with the following parameters:"
echo "Model Path: $MODEL_PATH"
echo "Tokenizer Path: $TOKENIZER_PATH"
echo "Benchmarks: $BENCHMARKS"

# Python 스크립트 실행
python run.py \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --benchmarks "$BENCHMARKS"
