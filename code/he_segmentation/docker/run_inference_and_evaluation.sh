#!/bin/bash

# Determine the directory where this script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR/../../.."

docker run --rm \
    -v "$(pwd)":/ignite_data_toolkit \
    -w /ignite_data_toolkit/code/he_segmentation/ \
    --entrypoint bash \
    --gpus all \
    ignite_data_toolkit:he \
    -c "python3 inference.py && python3 evaluation.py"