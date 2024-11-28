#!/bin/bash

# Define the target directory
TARGET_DIR=~/tmp

# Install Anaconda
echo "Downloading and installing Anaconda..."
cd "$TARGET_DIR"  # Go to the tmp directory

# Download the Anaconda installer
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-MacOSX-arm64.sh

# Verify the file
ls -lh Anaconda3-2024.10-1-MacOSX-arm64.sh

# Make it executable
chmod +x Anaconda3-2024.10-1-MacOSX-arm64.sh

# Install Anaconda
./Anaconda3-2024.10-1-MacOSX-arm64.sh

# Initialize Anaconda
#conda init

# Source .zshrc to apply changes
#source ~/.zshrc
