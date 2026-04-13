#!/bin/bash

echo "=== Install required packages ==="
pip install ai-models ai-models-fourcastnetv2 cdsapi attrs h5py numpy==1.26.4 matplotlib cartopy pygrib

echo "=== Create working directory ==="
mkdir -p fourcastnetv2
cd fourcastnetv2 || exit

echo "=== Check normalization files ==="

if [[ ! -f global_means.npy ]]; then
curl -L -O https://sites.ecmwf.int/repository/ai-models/test-data/fourcastnetv2/small/global_means.npy
fi

if [[ ! -f global_stds.npy ]]; then
curl -L -O https://sites.ecmwf.int/repository/ai-models/test-data/fourcastnetv2/small/global_stds.npy
fi

if [[ ! -f weights.tar ]]; then
curl -L -O https://sites.ecmwf.int/repository/ai-models/test-data/fourcastnetv2/small/weights.tar
fi

echo "Checking CDS API credentials..."

if [ -f "$HOME/.cdsapirc" ]; then
    echo "CDS API credentials already exist. Skipping setup."
else
    echo "CDS API credentials not found."

    read -p "Enter your CDS API Key: " CDS_KEY

    cat <<EOF > ~/.cdsapirc
url: https://cds.climate.copernicus.eu/api
key: ${CDS_KEY}
EOF

    chmod 600 ~/.cdsapirc
    echo "CDS API credentials saved."
fi

echo "Checking PyTorch version..."

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)

if [[ "$TORCH_VERSION" == 2.5.1* ]]; then
    echo "Correct PyTorch version already installed: $TORCH_VERSION"
else
    echo "Installing compatible PyTorch version..."
    pip install torch==2.5.1
fi

echo "=== Running prediction ==="

ai-models --input cds --date 20251002 --time 0000 fourcastnetv2-small

echo "✅ Forecast complete: fourcastnetv2-small.grib"
