# Fine-Tuning FourCastNet (FCN) on AWS EC2

This is a step-by-step tutorial to guide users through training and fine-tuning FourCastNet using an AWS EC2 instance. By the end of this tutorial, you will be able to:

- Launch an EC2 instance with GPU support
- Clone the FourCastNet repository
- Mount a Docker container with all required dependencies
- Download ERA5 reanalysis data from the Copernicus Climate Data Store (CDS)
- Convert netCDF files to HDF5 format
- Prepare normalization statistics
- Adjust configuration files for fine-tuning
- Train the model on new data
- Evaluate model performance and visualize predictions

---

## Step 1: Launch a GPU-Powered EC2 Instance

1. Go to [https://aws.amazon.com/ec2/](https://aws.amazon.com/ec2/)
2. Launch a new instance using the Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)
3. Choose an instance type with a GPU (e.g., `g4dn.xlarge`, `p3.2xlarge`)

    a. Ensure that the GPU has sufficient memory for training >16GB
4. Configure security group to allow SSH (port 22)
5. Allocate sufficient storage through AWS Elastic Block Store volume configuration (≥ 100 GB)
6. SSH into your instance:

```bash
ssh -i path/to/key.pem ec2-user@your-ec2-public-ip
```

---

## Step 2: Set Up Docker Environment

1. Install Docker:

```bash
sudo apt update && sudo apt install -y docker.io
sudo usermod -aG docker $USER
newgrp docker
```

2. Pull a prepared FourCastNet-compatible image (example):

```bash
docker pull henrylisdsu/nvidia-cuda-12.4:v2.5
```

3. Run the container:

```bash
docker run --gpus all -it --rm -v $PWD:/workspace -p 8888:8888 henrylisdsu/nvidia-cuda-12.4:v2.5 bash
```

---

## Step 3: Clone the FourCastNet Repository

```bash
git clone https://github.com/NVlabs/FourCastNet.git
cd FourCastNet
```

Install Python dependencies if needed:

```bash
pip install -r requirements.txt
pip install h5py xarray netCDF4 cdsapi
```

---

## Step 4: Download ERA5 Data from CDS

1. Create a CDS account: [https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
2. Create `~/.cdsapirc` with your credentials.
3. Use `cdsapi` to request data:

```python
import cdsapi
c = cdsapi.Client()
c.retrieve(
  'reanalysis-era5-single-levels',
  {
      'product_type': 'reanalysis',
      'variable': ['2m_temperature', '10m_u_component_of_wind'],
      'year': '2016',
      'month': ['01'],
      'day': ['01', '02'],
      'time': ['00:00', '06:00', '12:00', '18:00'],
      'format': 'netcdf'
  },
  'output.nc')
```

---

## Step 5: Convert netCDF to HDF5

```python
######PARALLEL CONVERT TO H5######


import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def load_variable(fpath, var):
    with h5py.File(fpath, "r") as f:
        arr = f[var][()]
        if arr.ndim == 4:
            arr = arr[:, 0, :, :]  # [time, ens, lat, lon] -> [time, lat, lon]
        return var, arr

def convert_to_fcn_target_shape(src_file, dest_file, var_list, total_channels=21):
    # First read all arrays in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda var: load_variable(src_file, var), var_list))

    # Determine dimensions
    _, first_arr = results[0]
    time_len, lat_len, lon_len = first_arr.shape
    print(f"Creating {dest_file} with shape: ({time_len}, {total_channels}, {lat_len}, {lon_len})")

    # Create output file and write sequentially
    with h5py.File(dest_file, "w") as out:
        fields = out.create_dataset("fields", (time_len, total_channels, lat_len, lon_len), dtype=np.float32)

        for i, (var, arr) in enumerate(results):
            fields[:, i, :, :] = arr

        if len(var_list) < total_channels:
            print(f"Filling remaining {total_channels - len(var_list)} channels with zeros")
            for i in range(len(var_list), total_channels):
                fields[:, i, :, :] = 0.0


convert_to_fcn_target_shape(
    "<your_cds_data>.nc",
    "<your_output_filename>.h5",
    ["u10", "v10", "t2m","msl","sp"]
)
```

---

## Step 6: Compute Global Mean and Std

Use a script to download pre-computed normalization stats:

```bash
curl -O https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_means.npy
curl -O https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_stds.npy
curl -O https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/weights.tar
```

---

## Step 7: Configure Training for Fine-Tuning

Create or modify a YAML config file:

```yaml
 ###SAMPLE WITH ONE YEAR###
mini_demo:
  <<: *FULL_FIELD
  in_channels: [0, 1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  out_channels: [0, 1 ,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  batch_size: 2
  max_epochs: 5
  prediction_length: 16
  n_initial_conditions: 1
  pretrained: !!bool True
  two_step_training: !!bool True
  pretrained_ckpt_path: '/path_to_checkpoint/backbone.ckpt'


  train_data_path: '/path_to_directory/28march'
  valid_data_path: '/path_to_directory/28march'
  inf_data_path: '/path_to_directory/28march'

  global_means_path: '/path_to_directory/global_means.npy'
  global_stds_path: '/path_to_directory/global_stds.npy'
  time_means_path: '/path_to_directory/time_means.npy'

  exp_dir: '/path_to_exp_dir'
  log_to_wandb: !!bool True
  save_checkpoint: !!bool True
  pretrained_ckpt_path: '/path_to_directory/backbone.ckpt'
```

---

## Step 8: Launch Training

```bash
python3 train.py --yaml_config=config/AFNO.yaml --config=mini_demo --run_num=0

```

Watch logs and monitor loss, validation metrics, and checkpoint saves.

---

## Step 9: Monitor Training with wandb

Once the training is launched you can monitor the progress of the training using wandb if you connected you account. This is highly recommended as wandb seemlessly synchronizes with your training model and will plot validation metrics as training epochs complete.

---

## Step 10: Visualize Predictions

You can use matplotlib or cartopy to plot model output.

```python
import matplotlib.pyplot as plt
plt.imshow(preds[0, 0], cmap="coolwarm")
plt.title("FCN Prediction")
plt.colorbar()
plt.show()
```

---

## Notes

- Do not install packages with `--user` inside Docker.
- Use `du -sh` to monitor disk usage if using limited EBS volumes.
- Store large files in `/workspace` or an attached volume.

---

## Questions?

Contact Iman Khadir at [[imankhadir@gmail.com](mailto\:imankhadir@gmail.com)] or visit the [FourCastNet GitHub](https://github.com/NVlabs/FourCastNet).
