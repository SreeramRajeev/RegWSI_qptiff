#!/bin/bash
# Fix PyVIPS and libvips installation issues

echo "Fixing PyVIPS installation..."

# 1. First, let's check what's installed
echo "Checking current libvips installation..."
ldconfig -p | grep vips || echo "No libvips found in ldconfig"

# 2. Remove problematic pyvips
pip uninstall -y pyvips

# 3. Install libvips properly
echo "Installing libvips from source (more reliable)..."
apt-get update
apt-get install -y \
    build-essential \
    pkg-config \
    glib2.0-dev \
    libexpat1-dev \
    libtiff5-dev \
    libjpeg-dev \
    libgsf-1-dev \
    libgif-dev \
    libwebp-dev

# Download and build libvips
cd /tmp
wget https://github.com/libvips/libvips/releases/download/v8.14.2/vips-8.14.2.tar.xz
tar xf vips-8.14.2.tar.xz
cd vips-8.14.2
./configure
make -j$(nproc)
make install
ldconfig

# 4. Set environment variables
echo "Setting environment variables..."
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# 5. Reinstall pyvips
echo "Reinstalling pyvips..."
pip install pyvips

# 6. Test the installation
echo ""
echo "Testing PyVIPS installation..."
python -c "
try:
    import pyvips
    print('✅ PyVIPS imported successfully')
    print(f'   Version: {pyvips.__version__}')
except Exception as e:
    print(f'❌ PyVIPS import failed: {e}')
"

# 7. Alternative: If above fails, try system package
if [ $? -ne 0 ]; then
    echo "Trying alternative installation method..."
    apt-get install -y python3-pyvips
fi

echo ""
echo "Fix completed. Please run 'source ~/.bashrc' or restart your terminal."
