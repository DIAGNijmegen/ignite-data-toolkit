#!/bin/bash

# Determine the directory where this script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Navigate to the project root directory
cd "$SCRIPT_DIR/../../.."

docker build -t ignite_data_toolkit:pdl1  -f "$SCRIPT_DIR/Dockerfile" .