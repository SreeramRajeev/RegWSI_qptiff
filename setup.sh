#!/bin/bash
# Setup script for vast.ai H200 environment
# Run this after connecting to your instance

echo "==================================="
echo "Setting up Registration Environment"
echo "==================================="

# Update system
echo "Updating system packages..."
apt-get update
apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    libvips \
    libvips-dev \
    openslide-tools \
    libopenslide-dev \
    libgdk-pixbuf2.0-dev \
    libffi-dev \
    libxml2-dev \
    libxslt-dev \
    libjpeg-dev \
    libtiff-dev

# Create workspace
echo "Creating workspace..."
mkdir -p /workspace/{data,output,temp}

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip

# Core packages
pip install \
    numpy \
    torch \
    torchvision \
    opencv-python \
    pillow \
    matplotlib \
    scikit-image \
    tifffile \
    SimpleITK

# Whole slide image packages
echo "Installing WSI packages..."
pip install \
    openslide-python \
    pyvips \
    large-image \
    large-image-source-openslide \
    large-image-source-tiff

# Install DeepHistReg
echo "Installing DeepHistReg..."
pip install deeperhistreg

# Verify GPU
echo ""
echo "Checking GPU..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Verify installations
echo ""
echo "Verifying installations..."
python -c "
import sys
libraries = {
    'numpy': 'NumPy',
    'torch': 'PyTorch', 
    'cv2': 'OpenCV',
    'PIL': 'Pillow',
    'tifffile': 'Tifffile',
    'SimpleITK': 'SimpleITK',
    'deeperhistreg': 'DeepHistReg',
    'openslide': 'OpenSlide',
    'pyvips': 'PyVIPS'
}

print('Library Status:')
print('-' * 40)
for module, name in libraries.items():
    try:
        __import__(module)
        print(f'✅ {name:<20} installed')
    except ImportError:
        print(f'❌ {name:<20} NOT installed')
"

echo ""
echo "Setup complete!"
echo ""
echo "Usage example:"
echo "python qptiff_registration.py \\"
echo "  --he-qptiff /workspace/data/HE.qptiff \\"
echo "  --if-qptiff /workspace/data/IF.qptiff \\"
echo "  --output-dir /workspace/output \\"
echo "  --resolution-level 1 \\"
echo "  --if-channels 0 1 5 \\"
echo "  --use-openslide  # Optional: for faster loading"
