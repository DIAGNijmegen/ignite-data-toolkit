#!/bin/bash

# Determine the directory where this script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Navigate to the project root directory
cd "$SCRIPT_DIR/../../.."

docker run --rm -it \
    -v "$(pwd)":/ignite_data_toolkit \
    -w /ignite_data_toolkit/code/pdl1_detection/ \
    --gpus all \
    ignite_data_toolkit:pdl1
