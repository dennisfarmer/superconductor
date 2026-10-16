#!/bin/bash

# Setup script for sc_env_atlas conda environment
# Usage: bash setup_env.sh

echo "Creating conda environment 'sc_env_atlas'..."
conda env create -f environment.yml

echo ""
echo "Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate sc_env_atlas"
echo ""
echo "To deactivate the environment, run:"
echo "  conda deactivate"
