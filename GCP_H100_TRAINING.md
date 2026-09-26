# Train pi0.5 on a GCP H100 VM

Use a prebuilt Google Deep Learning VM image with Ubuntu 24.04, CUDA 12.9,
and NVIDIA driver 580. Run your training container with `--gpus all`. Build the
container image only when code, dependencies, or bundled configs change. Copy datasets separately,
and save checkpoints under `/data` so they survive container replacement.

Run local commands from the LeRobot repository root (the directory containing
`pyproject.toml`, `gcp-startup.sh`, and `docker/`).

## 1. Set local variables

Set these in each new local shell:

```bash
export PROJECT_ID="extend-robotics-gcp-dev"
export REGION="europe-west2"
export ZONE="europe-west2-b"
export REPOSITORY="lerobot"
export VM_NAME="pi05-h100"
export MACHINE_TYPE="a3-highgpu-8g"
export VM_IMAGE_PROJECT="deeplearning-platform-release"
export VM_IMAGE_FAMILY="common-cu129-ubuntu-2404-nvidia-580"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/pi05-h100:v1"
```

## 2. One-time cloud setup

```bash
gcloud services enable compute.googleapis.com artifactregistry.googleapis.com \
  --project="${PROJECT_ID}"

gcloud auth configure-docker "${REGION}-docker.pkg.dev"
```

If the Artifact Registry repository does not already exist, create it once:

```bash
gcloud artifacts repositories create "${REPOSITORY}" \
  --project="${PROJECT_ID}" \
  --repository-format=docker \
  --location="${REGION}"
```

The VM service account needs `roles/artifactregistry.reader` on the repository.
The subnet `lerobot-europe-west2` and the firewall rule associated with
`allow-direct-ssh` must already be configured for your environment.

## 3. Build and push when the image changes

```bash
docker build --platform=linux/amd64 \
  -f docker/Dockerfile.pi05-h100-gcp \
  -t "${IMAGE_URI}" .

docker push "${IMAGE_URI}"
```

Keep `IMAGE_URI` in `gcp-startup.sh` aligned with the image you push. Dataset
updates do not require an image rebuild.

## 4. Create the VM once

The VM image supplies CUDA and the NVIDIA driver. The
[startup script](gcp-startup.sh) checks the preinstalled driver, installs Docker
and NVIDIA Container Toolkit only if missing, and configures Docker GPU access.
It authenticates to the registry using the VM service account, checks PyTorch on
every visible GPU, and starts `pi05-trainer` with `--gpus all` and persistent
storage mounted at `/data`. It does not install drivers or reboot the VM.

The host's CUDA 12.9 installation is separate from the container's CUDA 12.8
libraries. Training uses the container environment, so the Dockerfile needs no
change. The host driver supplies GPU access.

There is no GPU count or device list in the Docker commands. GCP still requires
a machine type: `MACHINE_TYPE` determines the physical GPUs attached to the VM.
The example uses a single-H100 machine; change this variable for another suitable
A3 machine type. Making all GPUs visible does not automatically make training
distributed; the training command below launches a single process.

The selected base image family is listed in
[Google's Deep Learning VM image catalog](https://docs.cloud.google.com/deep-learning-vm/docs/images).
Before creating the VM, resolve the current image to confirm availability:

```bash
gcloud compute images describe-from-family "${VM_IMAGE_FAMILY}" \
  --project="${VM_IMAGE_PROJECT}" \
  --format='value(name)'
```

For the Docker runtime configuration, see
[NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
gcloud compute instances create "${VM_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-family="${VM_IMAGE_FAMILY}" \
  --image-project="${VM_IMAGE_PROJECT}" \
  --no-shielded-secure-boot \
  --boot-disk-size=500GB \
  --subnet=lerobot-europe-west2 \
  --tags=allow-direct-ssh \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata-from-file=startup-script=gcp-startup.sh
```

For A3 provisioning constraints, see
[Google's A3 VM documentation](https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-accelerator-optimized).

For subsequent sessions, start the existing stopped VM:

```bash
gcloud compute instances start "${VM_NAME}" \
  --project="${PROJECT_ID}" --zone="${ZONE}"
```

To inspect setup progress:

```bash
gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT_ID}" --zone="${ZONE}" \
  --command='sudo journalctl -u google-startup-scripts.service -n 100 --no-pager'
```

Allow startup to finish, then check that the container is running:

```bash
gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT_ID}" --zone="${ZONE}" \
  --command='sudo docker ps --filter name=pi05-trainer'
```

## 5. Copy the dataset when it changes

```bash
set -o pipefail
tar -C datasets -cf - leyland_merged_500 | \
  gcloud compute ssh "${VM_NAME}" \
    --project="${PROJECT_ID}" --zone="${ZONE}" \
    --command="sudo docker exec -i pi05-trainer sh -c \
      'mkdir -p /data/datasets && tar --no-same-owner -xf - -C /data/datasets'"
```

This writes to the VM's mounted storage, not the image. Repeating this command
overwrites matching files but does not remove files deleted from the local
dataset. Avoid updating a dataset while training reads it.

## 6. Provide runtime credentials

Make Hugging Face and W&B credentials available before training. Do not put
tokens in the Dockerfile, image, or committed files.

The startup script automatically loads `/etc/lerobot/runtime.env` if it exists.
Create it on the VM using your secret-management workflow, owned by root with
mode `0600`, containing `HF_TOKEN=...` and `WANDB_API_KEY=...` as needed. If you
create or change it after startup, rerun the startup script before training:

```bash
gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT_ID}" --zone="${ZONE}" \
  --command='sudo google_metadata_script_runner startup'
```

Rerunning startup recreates the container and stops any running training process.
The environment file is read by Docker; it is never baked into the image.

## 7. Launch training without tmux

Use a new output directory and log filename for each fresh run:

```bash
gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT_ID}" --zone="${ZONE}" \
  --command="sudo docker exec -d pi05-trainer bash -c '
    mkdir -p /data/logs
    exec /opt/lerobot/.venv/bin/accelerate launch \
      --multi_gpu \
      --num_processes=8 \
      /opt/lerobot/.venv/bin/lerobot-train \
      --config_path=/opt/lerobot/configs/pi05_cloud_leyland_merged.json \
      --dataset.root=/data/datasets/leyland_merged_500 \
      --output_dir=/data/outputs/pi05_leyland_run \
      > /data/logs/pi05_leyland_run.log 2>&1
  '"
```

The SSH command returns after launching the process; it does not report whether
training later succeeds. Check the log:

```bash
gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT_ID}" --zone="${ZONE}" \
  --command='sudo docker exec pi05-trainer tail -n 100 -f /data/logs/pi05_leyland_run.log'
```

Press Ctrl-C to stop following the log. Training continues after SSH disconnects.

For an interactive container shell:

```bash
gcloud compute ssh "${VM_NAME}" \
  --project="${PROJECT_ID}" --zone="${ZONE}" \
  --command='sudo docker exec -it pi05-trainer bash' -- -t
```

## Persistence and Spot interruptions

- `/data` maps to `/srv/lerobot` on the VM boot disk. Datasets,
  caches, logs, and outputs stored there survive container replacement and VM
  stop/start. This is not a backup; deleting the boot disk deletes this data.
- The config's default relative `output_dir` is inside the container. Always
  override it with a path under `/data/outputs`, as above.
- A Spot interruption stops training. Docker's `--restart=unless-stopped`
  restarts the container's `sleep infinity` command, not a training process
  launched with `docker exec`. Restart the VM and explicitly resume from a saved
  checkpoint using LeRobot's resume options.
- After training finishes, stop the VM to release the running GPU allocation:

```bash
gcloud compute instances stop "${VM_NAME}" \
  --project="${PROJECT_ID}" --zone="${ZONE}"
```

The normal workflow is: start VM, copy the dataset if changed, launch training,
follow the log, then stop the VM.
