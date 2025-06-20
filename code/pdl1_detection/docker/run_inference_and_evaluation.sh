#!/bin/bash

# Determine the directory where this script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Navigate to the project root directory
cd "$SCRIPT_DIR/../../.."

docker run --rm \
    -v "$(pwd)":/ignite_data_toolkit \
    -w /ignite_data_toolkit/code/pdl1_detection/ \
    --network host \
    --entrypoint bash \
    --gpus all \
    ignite_data_toolkit:pdl1 \
    -c "python3.11 nuclei_test_set_inference.py && python3.11 pdl1_test_set_inference.py"