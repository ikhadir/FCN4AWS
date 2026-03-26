#!/bin/bash

# === Step 6: Create FourCastNetV2 Directory ===
echo "Creating directory: fourcastnetv2"
mkdir -p fourcastnetv2
cd fourcastnetv2 || exit

# === Step 7: Set Up CDS API Credentials ===
echo "Setting up CDS API credentials..."
cat <<EOF > ~/.cdsapirc
url: https://cds.climate.copernicus.eu/api
key: <your-uid>:<your-api-key>
EOF
chmod 600 ~/.cdsapirc

# === Step 7.5: Install CDS API Python Client ===
echo "Installing CDS API Python client..."
pip install "cdsapi>=0.7.4"
pip install --upgrade attrs

# === Step 8: Download FCNv2 Normalization Files ===
echo "Downloading FCNv2 normalization files..."
curl -L -O -A "Mozilla/5.0" https://sites.ecmwf.int/repository/ai-models/test-data/fourcastnetv2/small/global_means.npy
curl -L -O -A "Mozilla/5.0" https://sites.ecmwf.int/repository/ai-models/test-data/fourcastnetv2/small/global_stds.npy
curl -L -O -A "Mozilla/5.0" https://sites.ecmwf.int/repository/ai-models/test-data/fourcastnetv2/small/weights.tar

# === Step 9: Install ECMWF AI Model Packages ===
echo "Installing ECMWF AI model packages..."
pip install ai-models
pip install ai-models-fourcastnetv2

echo "✅ Setup complete."

