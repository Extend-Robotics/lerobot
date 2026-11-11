#!/bin/bash
# EC2 user data installer and per-boot container setup. See docs/extend/AWS_H100_TRAINING.md.
set -euo pipefail

if [[ "${1:-}" != --run ]]; then
  : "${AWS_REGION:?Set AWS_REGION in user data}"
  : "${IMAGE_URI:?Set IMAGE_URI in user data}"
  install -d -m 0700 /etc/lerobot
  printf 'AWS_REGION=%q\nIMAGE_URI=%q\n' "$AWS_REGION" "$IMAGE_URI" > /etc/lerobot/aws.conf
  chmod 0600 /etc/lerobot/aws.conf
  install -m 0755 "$0" /usr/local/sbin/lerobot-aws-startup
  cat > /etc/systemd/system/lerobot-container.service <<'UNIT'
[Unit]
Description=Prepare the LeRobot GPU training container
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/lerobot-aws-startup --run
RemainAfterExit=yes
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable --now lerobot-container.service
  exit 0
fi

source /etc/lerobot/aws.conf
export AWS_REGION AWS_DEFAULT_REGION="$AWS_REGION"
export DEBIAN_FRONTEND=noninteractive
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_ID" != 24.04 ]]; then
  echo "Use the Ubuntu 24.04 x86_64 Deep Learning Base GPU AMI." >&2
  exit 1
fi
# Keep the AMI's driver. The container supplies CUDA userspace libraries.
nvidia-smi
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
if ! command -v aws >/dev/null; then
  if [[ ! -x /opt/lerobot-awscli/bin/aws ]]; then
    apt-get update
    apt-get install -y -o DPkg::Lock::Timeout=300 python3-venv
    python3 -m venv /opt/lerobot-awscli
    /opt/lerobot-awscli/bin/pip install awscli
  fi
  export PATH="/opt/lerobot-awscli/bin:$PATH"
fi
nvidia-ctk runtime configure --runtime=docker
systemctl enable docker
systemctl restart docker

# This directory is on the EBS root disk, never the AMI's local NVMe scratch disk.
install -d -o 1000 -g 1000 /srv/lerobot
REGISTRY_HOST="${IMAGE_URI%%/*}"
DOCKER_CONFIG=$(mktemp -d)
export DOCKER_CONFIG
trap 'rm -rf "$DOCKER_CONFIG"' EXIT
# Use the EC2 instance role; do not enable shell tracing or embed AWS keys.
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "$REGISTRY_HOST"
docker pull "$IMAGE_URI"
docker run --rm --gpus all "$IMAGE_URI" python -c '
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
  --volume=/srv/lerobot:/data \
  "$IMAGE_URI" sleep infinity
