#!/bin/bash

# This script sets up the complete toy EDM environment

echo "=========================================="
echo "edm-toy setup"
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv edm_env

# Activate virtual environment
echo "Activating virtual environment..."
source edm_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -e .

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo ""
echo "To activate the environment:"
echo "source edm_env/bin/activate"
echo ""
echo "Documentation:"
echo "- README.md"
echo ""