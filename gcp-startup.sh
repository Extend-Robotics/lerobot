#!/bin/bash
set -euo pipefail

# GCE runs startup scripts as root. Use the Ubuntu 24.04 Deep Learning VM
# base image documented in GCP_H100_TRAINING.md (CUDA and driver preinstalled).
source /etc/os-release
if [[ "${ID}" != ubuntu || "${VERSION_ID}" != 24.04 ]]; then
  echo "This startup script requires Ubuntu 24.04." >&2
  exit 1
fi

REGISTRY_HOST="europe-west2-docker.pkg.dev"
IMAGE_URI="${REGISTRY_HOST}/extend-robotics-gcp-dev/lerobot/pi05-h100:v1"
DATA_DIR="/srv/lerobot"
export DEBIAN_FRONTEND=noninteractive

# Use the image's driver; do not install or replace it during startup.
if ! nvidia-smi; then
  echo "Preinstalled GPU driver is unavailable. Check the VM image and boot logs." >&2
  exit 1
fi

# Keep any Docker installation provided by the image.
if ! command -v docker >/dev/null; then
  apt-get update
  apt-get install -y -o DPkg::Lock::Timeout=300 docker.io
fi
if ! command -v nvidia-ctk >/dev/null; then
  apt-get update
  apt-get install -y -o DPkg::Lock::Timeout=300 ca-certificates curl gnupg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y -o DPkg::Lock::Timeout=300 nvidia-container-toolkit
fi

nvidia-ctk runtime configure --runtime=docker
systemctl enable docker
systemctl restart docker

mkdir -p "${DATA_DIR}"
chown 1000:1000 "${DATA_DIR}"

# Use the VM service account and an ephemeral Docker config. Never enable xtrace.
DOCKER_CONFIG=$(mktemp -d)
export DOCKER_CONFIG
trap 'rm -rf "${DOCKER_CONFIG}"' EXIT
curl -fsS --retry 5 -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token | \
  python3 -c 'import json, sys; print(json.load(sys.stdin)["access_token"])' | \
  docker login -u oauth2accesstoken --password-stdin "https://${REGISTRY_HOST}"
docker pull "${IMAGE_URI}"

# Verify every visible GPU without specifying a GPU count or device list.
docker run --rm --gpus all "${IMAGE_URI}" python -c '
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
for index in range(torch.cuda.device_count()):
    x = torch.ones(16, device=f"cuda:{index}")
    assert x.sum().item() == 16
    torch.cuda.synchronize(index)
    print("GPU verified:", index, torch.cuda.get_device_name(index))
'

ENV_ARGS=()
if [[ -f /etc/lerobot/runtime.env ]]; then
  ENV_ARGS=(--env-file=/etc/lerobot/runtime.env)
fi

if docker container inspect pi05-trainer >/dev/null 2>&1; then
  docker rm -f pi05-trainer
fi

docker run -d \
  --name pi05-trainer \
  --restart=unless-stopped \
  --gpus all \
  "${ENV_ARGS[@]}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume="${DATA_DIR}:/data" \
  "${IMAGE_URI}" \
  sleep infinity
