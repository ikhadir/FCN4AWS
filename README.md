# San Diego State University Climate Informatics Lab (SCIL) - FCN4AWS

A user guide for generating AI weather forecasts using AWS

AI weather predictions have now become more popular as seen by a number of data driven weather forecasting models such as FourCastNet, GenCast, GraphCast, Aurora, and Pangu-Weather. These models have greatly reduced the need for extremely large supercomputing resources and have provided researchers the opportunity to make weather predictions with access to just one GPU. San Diego State University Climate Informatics Lab has already demonstrated the ability of a small research group, in collaboration with university IT department and NVIDIA scientists, to produce high quality AI weather predictions. A user's guide has been created on GitHub to replicate our work.

SCIL provides this user guide which will be primarily useful to research groups and weather enthusiasts who would like to produce high quality global weather forecasts and do not have access to a local or shared GPU cluster. Our approach is then to determine the computational resources required and services which can be provided by cloud computing platforms such as Amazon Web Services.

---

## Step-by-Step Guide to Run FCNv2 Forecasts on AWS

### 1. vCPU Quota Request (for New Accounts)

If your AWS account is new, your default vCPU quota is likely zero. You will need to request an increase:

* **Service**: Amazon Elastic Compute Cloud (Amazon EC2)
* **Quota Name**: Running On-Demand G and VT instances
* **Request**: Increase to at least **4 vCPUs** at the **Account Level**

### 2. Launch the Deep Learning AMI with GPU Support

* Open the AMI: **Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Amazon Linux 2023)** — version `20250607`
* This AMI comes with the **NVIDIA driver preinstalled** ✅

### 3. Install Python pip (if not already installed)

```bash
sudo yum install python3-pip -y
```

### 4. Install PyTorch and CUDA Toolkit

```bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### 5. Verify CUDA with `chk_cuda.py`

Use a simple script to verify:

```python
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

### 6. Prepare FourCastNet Directory

```bash
mkdir fourcastnetv2
```

### 7. Set Up CDS API Credentials

Create a `.cdsapirc` file in your home directory:

```bash
nano ~/.cdsapirc
```

Paste your credentials from [https://cds.climate.copernicus.eu/how-to-api](https://cds.climate.copernicus.eu/how-to-api):

```ini
url: https://cds.climate.copernicus.eu/api/v2
key: <your-uid>:<your-api-key>
```

Install the required Python client:

```bash
pip install "cdsapi>=0.7.4"
pip install --upgrade attrs
```

### 8. Download Precomputed FCNv2 Normalization Files in the fourcastnetv2 Directory

```bash
curl -O https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_means.npy
curl -O https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_stds.npy
curl -O https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/weights.tar
```

### 9. Install the ECMWF AI Models Package

```bash
pip install ai-models
pip install ai-models-fourcastnetv2
```

### 10. Generate a Forecast

You’re now ready to generate a forecast using FCNv2:

```bash
ai-models --input cds --date 20230110 --time 0000 fourcastnetv2-small
```

You should see output confirming that the model is using CUDA:

```
INFO Using device 'CUDA'. The speed of inference depends greatly on the device.
```

This completes your FCNv2 setup using AWS!
