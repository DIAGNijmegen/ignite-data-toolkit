#!/bin/bash

# Determine the directory where this script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Navigate to the project root directory
cd "$SCRIPT_DIR/../../.."

docker run --rm -it \
    -v "$(pwd)":/ignite_data_toolkit \
    -w /ignite_data_toolkit/code/he_segmentation/ \
    --gpus all \
    ignite_data_toolkit
