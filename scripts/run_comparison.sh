#!/usr/bin/env bash
# Run full comparison of TF-IDF and Embeddings models
# This script runs all make targets in the correct order

set -e

echo "======================================================================"
echo "CRS Model Comparison"
echo "======================================================================"
echo ""

# Run the comparison using make
make compare

echo ""
echo "======================================================================"
echo "Comparison Complete!"
echo "======================================================================"
