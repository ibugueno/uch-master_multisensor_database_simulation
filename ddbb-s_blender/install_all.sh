#!/bin/bash

# Define the target directory
TARGET_DIR=~/tmp

# Create tmp directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Install Homebrew (download and install in tmp)
echo "Installing Homebrew..."
cd "$TARGET_DIR"  # Cambiar al directorio tmp
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Update .zprofile for Homebrew (if needed)
echo "Updating .zprofile for Homebrew..."
echo >> ~/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
source ~/.zprofile

# Install tmux using Homebrew
echo "Installing tmux..."
brew install tmux

# Install Blender (download and store in tmp)
echo "Downloading and installing Blender..."
cd "$TARGET_DIR"  # Asegurarse de que estamos en tmp antes de descargar
curl -L -O https://download.blender.org/release/Blender4.0/blender-4.0.2-macos-arm64.dmg

# Mount Blender .dmg
hdiutil mount blender-4.0.2-macos-arm64.dmg

# Copy Blender.app to Applications
cp -R /Volumes/Blender/Blender.app /Applications/

# Unmount Blender .dmg
hdiutil unmount /Volumes/Blender

# Check Blender version
/Applications/Blender.app/Contents/MacOS/Blender --version

echo "Installation complete!"
