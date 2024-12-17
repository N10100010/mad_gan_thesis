#!/bin/bash

# Initialize Conda
conda init

# Step 1: Create a new conda environment
echo "Creating conda environment with Python 3.8..."
conda create -y --prefix ./mad_gan_env python=3.8.19

# Step 2: Activate the environment
echo "Activating the environment..."
conda activate ./mad_gan_env

# Step 3: Install pip packages
# Uncomment and provide a requirements.txt file if pip packages are needed
# echo "Installing pip packages..."
# pip install -r requirements.txt

# Step 4: Install Conda packages from environment.yml
if [ -f "environment.yml" ]; then
  echo "Installing Anaconda dependencies..."
  conda env update --prefix ./mad_gan_env --file environment.yml --prune
fi

echo "Environment setup complete!"
