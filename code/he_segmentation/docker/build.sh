#!/bin/bash

# Determine the directory where this script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Navigate to the project root directory
cd "$SCRIPT_DIR/../../.."

# Build the Docker image
docker build -t ignite_data_toolkit:he -f "$SCRIPT_DIR/Dockerfile" .
