# San Diego State University Climate Informatics Lab (SCIL) - FCN4AWS

A user guide for generating AI weather forecasts using AWS

AI weather predictions have now become more popular as seen by a number of data driven weather forecasting models such as FourCastNet, GenCast, GraphCast, Aurora, and Pangu-Weather. These models have greatly reduced the need for extremely large supercomputing resources and have provided researchers the opportunity to make weather predictions with access to just one GPU. San Diego State University Climate Informatics Lab has already demonstrated the ability of a small research group, in collaboration with university IT department and NVIDIA scientists, to produce high quality AI weather predictions. A user's guide has been created on GitHub to replicate our work.

SCIL provides this user guide which will be primarily useful to research groups and weather enthusiasts who would like to produce high quality global weather forecasts and do not have access to a local or shared GPU cluster. Our approach is then to determine the computational resources required and services which can be provided by cloud computing platforms such as Amazon Web Services.

---

## Step-by-Step Guide: Running FourCastNet on AWS

### 1. Launch a GPU-enabled EC2 Instance

* Go to the [AWS EC2 Console](https://console.aws.amazon.com/ec2/)
* Choose an AMI (Amazon Machine Image): **Deep Learning AMI (Amazon Linux 2)**
* Instance type: **g4dn.xlarge** or better
* Storage: Allocate at least **50 GB** for Docker image and model files
* Security Group: Open port **22** for SSH

### 2. SSH into Your Instance

```bash
ssh -i "your-key.pem" ec2-user@your-ec2-public-ip
```

### 3. Start and Enable Docker

```bash
sudo systemctl start docker
sudo usermod -aG docker ec2-user
newgrp docker
```

### 4. Pull the Prebuilt CUDA 12.4 Docker Image

```bash
docker pull henrylisdsu/nvidia-cuda-12.4:v2.5
```

### 5. Run the Container with GPU Access

```bash
docker run --gpus all -it henrylisdsu/nvidia-cuda-12.4:v2.5 /bin/bash
```

### 6. Install Any Missing Tools Inside the Container

If needed, switch to root and install tools:

```bash
apt update && apt install curl nano -y
```

### 7. Download FourCastNet Model Files

```bash
curl -O https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_means.npy
curl -O https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_stds.npy
curl -O https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/weights.tar
```

### 8. Install and Configure `ai-models` Package

Install required Python packages:

```bash
pip install torch==2.5.1 torchvision==0.18.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install ai-models ai-models-fourcastnetv2
```

### 9. Set Up Your CDS API Credentials

Create a `.cdsapirc` file in your home directory:

```ini
url: https://cds.climate.copernicus.eu/api/v2
key: your-uid:your-api-key
```

### 10. Run FourCastNet Inference

```bash
ai-models --input cds --date 20230110 --time 0000 fourcastnetv2-small
```

You should see:

```
INFO Using device 'CUDA'. The speed of inference depends greatly on the device.
```

And results will be saved as a `.grib` file.

---

This setup demonstrates how small research groups can replicate advanced climate AI workflows on cloud infrastructure, empowering broader participation in climate modeling and forecasting.
