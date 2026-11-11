# Train pi0.5 on an AWS H100 VM

This is the AWS equivalent of [the GCP workflow](GCP_H100_TRAINING.md):
launch an EC2 GPU VM, pull your custom Docker image from Amazon ECR, copy the
local dataset, and train with persistent checkpoints. You keep SSH access and
control of the host. This workflow uses EC2 directly.

The default is `p5.48xlarge`: **8 H100 GPUs, 80 GB each**, matching the GCP
`a3-highgpu-8g` and the eight-process training command. The GCP guide's prose
about one GPU is inconsistent with its actual machine type and launch command.
AWS also offers `p5.4xlarge` with one H100; see the single-GPU note below.
See [AWS P5 specifications](https://aws.amazon.com/ec2/instance-types/p5/).

Run local commands in Bash from the LeRobot repository root, containing
`pyproject.toml`, `aws-startup.sh`, and `docker/`. Have AWS CLI v2, Docker with
Buildx, SSH, and an authenticated AWS session available locally. Commands below
create billable resources when you run them. This guide does not assume a fixed
hourly price: check the selected region's On-Demand or Spot rate before launch.

## 1. Set local variables and select a network

```bash
export AWS_REGION="eu-west-2"
export AWS_DEFAULT_REGION="$AWS_REGION"
export AWS_PAGER=""
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPOSITORY="lerobot"
export VM_NAME="pi05-h100"
export INSTANCE_TYPE="p5.48xlarge"
export ROLE_NAME="lerobot-ec2-ecr-reader"
export INSTANCE_PROFILE_NAME="lerobot-ec2-ecr-reader"
export REGISTRY_HOST="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
export IMAGE_URI="${REGISTRY_HOST}/${REPOSITORY}:pi05-h100-v1"

# Replace with resources in this region and the same VPC.
export SUBNET_ID="subnet-REPLACE_ME"
export SECURITY_GROUP_ID="sg-REPLACE_ME"
export KEY_NAME="REPLACE_ME"
export SSH_KEY="$HOME/.ssh/REPLACE_ME.pem"
```

Use an existing EC2 key pair whose private key you have locally. The security
group must allow inbound TCP 22 from your public IP and outbound traffic for
ECR, package repositories, Hugging Face, and W&B. This guide uses a public IPv4
address: the subnet needs a route to an internet gateway and compatible network
ACLs. A private subnet requires a different SSH access path and outbound NAT or
appropriate endpoints. Public IPv4, EBS, and data transfer can incur charges.

Check that your instance type is offered in the subnet's Availability Zone:

```bash
export AZ=$(aws ec2 describe-subnets --subnet-ids "$SUBNET_ID" \
  --query 'Subnets[0].AvailabilityZone' --output text)
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=${INSTANCE_TYPE}" "Name=location,Values=${AZ}" \
  --query 'InstanceTypeOfferings[].InstanceType' --output text
```

An empty result means choose another supported subnet/AZ/region. An offering
is not a capacity guarantee. Check EC2 Service Quotas for **Running On-Demand P
instances** or **All P Spot Instance Requests**, as applicable. Quotas are in
vCPUs: `p5.48xlarge` requires 192. Quota approval does not reserve GPU capacity.

## 2. Create ECR and an EC2 instance role once

Skip repository creation if it already exists:

```bash
aws ecr create-repository --repository-name "$REPOSITORY" \
  --image-scanning-configuration scanOnPush=true
```

Your local AWS identity needs ECR push permissions, permission to create the IAM
resources below, EC2 launch permissions, `iam:PassRole` for this role, and
permission to read the public AMI parameter in step 4. If your organization
manages IAM centrally, use an existing equivalent role and instance profile.
The VM needs only ECR pull permissions for this workflow.

```bash
cat > /tmp/lerobot-ec2-trust.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
JSON

cat > /tmp/lerobot-ecr-pull.json <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"],
      "Resource": "arn:aws:ecr:${AWS_REGION}:${ACCOUNT_ID}:repository/${REPOSITORY}"
    }
  ]
}
JSON

aws iam create-role --role-name "$ROLE_NAME" \
  --assume-role-policy-document file:///tmp/lerobot-ec2-trust.json
aws iam put-role-policy --role-name "$ROLE_NAME" \
  --policy-name PullLeRobotImage \
  --policy-document file:///tmp/lerobot-ecr-pull.json
aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME"
aws iam add-role-to-instance-profile \
  --instance-profile-name "$INSTANCE_PROFILE_NAME" --role-name "$ROLE_NAME"
```

Allow time for IAM propagation before launch. If using existing resources,
skip creation and ensure the profile contains the correct role.
See [ECR authentication](https://docs.aws.amazon.com/AmazonECR/latest/userguide/registry_auth.html).

## 3. Build and push when the image changes

The existing Dockerfile works on AWS despite its `-gcp` filename. Its CUDA 12.8
libraries run inside the container; the host supplies a compatible NVIDIA driver
and NVIDIA Container Toolkit. No cloud-specific Dockerfile change is required.

```bash
set -o pipefail
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "$REGISTRY_HOST"
docker build --platform=linux/amd64 \
  -f docker/Dockerfile.pi05-h100-gcp -t "$IMAGE_URI" .
docker push "$IMAGE_URI"
```

Use a new tag for each changed image. Dataset updates do not require rebuilding.

## 4. Resolve the GPU AMI and prepare startup

Use the AWS Deep Learning Base OSS NVIDIA Driver GPU AMI, Ubuntu 24.04,
x86_64. Resolve its regional ID rather than copying an AMI ID from another region:

```bash
export AMI_ID=$(aws ssm get-parameter \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
  --query Parameter.Value --output text)
export ROOT_DEVICE=$(aws ec2 describe-images --image-ids "$AMI_ID" \
  --query 'Images[0].RootDeviceName' --output text)
aws ec2 describe-images --image-ids "$AMI_ID" \
  --query 'Images[0].{Name:Name,Architecture:Architecture,Root:RootDeviceName}'
```

Check the selected AMI's release notes for P5 support and a driver compatible
with the container's locked CUDA 12.8 PyTorch wheels (at least 570.86, per this
repository's Dockerfile). The host's CUDA toolkit version need not match the
container. See the [AMI catalog and release notes](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html).

Generate user data with the region and image URI, then append the checked-in
[startup script](../../aws-startup.sh):

```bash
{
  printf '#!/bin/bash\nexport AWS_REGION=%q\nexport IMAGE_URI=%q\n' "$AWS_REGION" "$IMAGE_URI"
  cat aws-startup.sh
} > /tmp/lerobot-aws-user-data.sh

cat > /tmp/lerobot-ebs.json <<JSON
[
  {
    "DeviceName": "${ROOT_DEVICE}",
    "Ebs": {
      "VolumeSize": 500,
      "VolumeType": "gp3",
      "Encrypted": true,
      "DeleteOnTermination": false
    }
  }
]
JSON
```

User data installs a `lerobot-container.service` systemd unit that runs on every
boot. The service checks the preinstalled driver, installs Docker and NVIDIA
Container Toolkit if missing, authenticates to ECR with the instance role,
checks PyTorch on every GPU, and creates `pi05-trainer`. An isolated AWS CLI
installation is added only if the AMI lacks `aws`.

`/data` in the container maps to `/srv/lerobot` on the **EBS root disk**. The
startup script never moves this directory onto the instance's local NVMe storage.
The runtime credentials file, if present, is loaded at container creation.

EC2 user data normally runs only on first launch; the systemd unit supplies the
subsequent-boot behavior. See [EC2 user data](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html).

## 5. Create the VM once

This command uses **On-Demand** capacity for a straightforward stop/start workflow:

```bash
export INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --count 1 \
  --key-name "$KEY_NAME" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SECURITY_GROUP_ID" \
  --associate-public-ip-address \
  --iam-instance-profile "Name=${INSTANCE_PROFILE_NAME}" \
  --metadata-options 'HttpTokens=required,HttpEndpoint=enabled' \
  --block-device-mappings file:///tmp/lerobot-ebs.json \
  --user-data file:///tmp/lerobot-aws-user-data.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${VM_NAME}}]" \
  --query 'Instances[0].InstanceId' --output text)
printf '%s\n' "$INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"
```

Record `INSTANCE_ID` for later sessions. Do not rerun `run-instances` to restart
an existing VM. If launch reports an invalid instance profile immediately after
IAM setup, allow propagation and retry. Capacity and quota errors require
resolving those conditions before retrying.

For **Spot**, add this argument to the same `run-instances` command *before*
launching, instead of creating the On-Demand instance above:

```bash
--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}'
```

The persistent request and `stop` behavior preserve the EBS-backed VM on
interruption. AWS restarts an AWS-interrupted stopped instance when matching
capacity returns; you cannot force that restart yourself. Spot requests can
remain active and incur compute charges when capacity returns. Inspect the
request as well as the instance when retiring resources. See
[Spot interruption behavior](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/interruption-behavior.html).

## 6. Connect and check startup

Run this again after every stop/start because the public IP can change:

```bash
export VM_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
chmod 0600 "$SSH_KEY"
ssh -i "$SSH_KEY" "ubuntu@${VM_IP}" \
  'sudo tail -n 100 /var/log/cloud-init-output.log; sudo journalctl -u lerobot-container.service -n 100 --no-pager'
```

SSH may need a little more time after the EC2 status checks pass. Allow image
pull and GPU checks to finish, then verify:

```bash
ssh -i "$SSH_KEY" "ubuntu@${VM_IP}" \
  'sudo systemctl status lerobot-container.service --no-pager; sudo docker ps --filter name=pi05-trainer'
```

If setup fails, inspect those logs, fix the cause, and retry:

```bash
ssh -i "$SSH_KEY" "ubuntu@${VM_IP}" \
  'sudo systemctl restart lerobot-container.service'
```

Restarting this service restarts Docker and recreates the training container,
stopping any training in it. Use a dedicated VM for this workflow.

## 7. Copy the dataset when it changes

```bash
set -o pipefail
tar -C datasets -cf - leyland_merged_500 | \
  ssh -i "$SSH_KEY" "ubuntu@${VM_IP}" \
    "sudo docker exec -i pi05-trainer sh -c \
      'mkdir -p /data/datasets && tar --no-same-owner -xf - -C /data/datasets'"
```

The dataset stays on EBS across container replacement and stop/start. Repeating
this command overwrites matching files but does not remove files deleted locally.
Do not update the dataset while training reads it. Check `df -h /srv/lerobot`
on the VM; images, datasets, caches, and checkpoints all share the 500 GB disk.
Increase the volume size before launch if needed.

## 8. Provide runtime credentials

Create `/etc/lerobot/runtime.env` on the VM using your secret-management workflow,
owned by root with mode `0600`. Include `HF_TOKEN=...` and `WANDB_API_KEY=...`
as needed by the policy and the config's enabled W&B integration. Do not commit
these values or put them in the Docker image or EC2 user data.

For manual setup, open an SSH session and run:

```bash
sudo install -d -m 0700 /etc/lerobot
sudo touch /etc/lerobot/runtime.env
sudo chmod 0600 /etc/lerobot/runtime.env
sudo chown root:root /etc/lerobot/runtime.env
sudo nano /etc/lerobot/runtime.env
sudo systemctl restart lerobot-container.service
```

The service restart loads the updated credentials and stops any existing
training process. The file survives stop/start on the EBS root disk.

## 9. Launch training without tmux

Run locally. Use a new output directory and log filename for each fresh run:

```bash
ssh -i "$SSH_KEY" "ubuntu@${VM_IP}" \
  "sudo docker exec -d pi05-trainer bash -c '
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

All eight GPUs are visible through `--gpus all`; Accelerate's eight processes
actually distribute training. The SSH command returning successfully only means
the detached command was launched. Follow the log to check training:

```bash
ssh -i "$SSH_KEY" "ubuntu@${VM_IP}" \
  'sudo docker exec pi05-trainer tail -n 100 -f /data/logs/pi05_leyland_run.log'
```

Ctrl-C stops log following; training continues after SSH disconnects. For a shell:

```bash
ssh -t -i "$SSH_KEY" "ubuntu@${VM_IP}" \
  'sudo docker exec -it pi05-trainer bash'
```

For **one H100**, choose `p5.4xlarge` before launch and remove `--multi_gpu`,
using `--num_processes=1`. The same config is a starting point, but memory use,
batch size, and throughput must be checked for that smaller setup.

## 10. Stop, restart, and resume

Stop the instance when you finish training:

```bash
aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"
```

Stopping ends running training. Save a checkpoint first if you need the current
state. EBS and other retained resources still incur charges while stopped.

For a later **On-Demand** session:

```bash
aws ec2 start-instances --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"
export VM_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
```

Check the startup service and container again. The service starts `sleep infinity`
in the container; it does **not** automatically resume training. For an interrupted
run with a complete checkpoint, use its saved configuration and `--resume=true`:

```bash
ssh -i "$SSH_KEY" "ubuntu@${VM_IP}" \
  "sudo docker exec -d pi05-trainer bash -c '
    mkdir -p /data/logs
    exec /opt/lerobot/.venv/bin/accelerate launch \
      --multi_gpu --num_processes=8 \
      /opt/lerobot/.venv/bin/lerobot-train \
      --config_path=/data/outputs/pi05_leyland_run/checkpoints/last/pretrained_model/train_config.json \
      --resume=true \
      >> /data/logs/pi05_leyland_run_resume.log 2>&1
  '"
```

Verify that this checkpoint path exists and was fully saved before resuming.
Keep the original GPU count and training settings for consistent recovery.
With the current config, checkpoints are saved every 5,000 steps; an interruption
can lose progress since the last completed save. Shorten `save_freq` if needed,
accounting for checkpoint time and disk space.

## Persistence and image updates

- `/data` is on EBS, so datasets, caches, logs, outputs, and checkpoints survive
  container replacement and VM stop/start. The container's writable layer is
  replaced on each boot; put persistent files under `/data`.
- `DeleteOnTermination=false` retains the root EBS volume if the instance is
  terminated. Record its volume ID and take EBS snapshots or copy outputs to S3
  for backup. A retained volume is not automatically attached to a replacement VM.
- P5's included local NVMe storage is temporary and loses data on stop/termination.
  See [EC2 storage](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Storage.html).
- For a new image, build and push a new tag, edit `IMAGE_URI` in the VM's
  `/etc/lerobot/aws.conf` with `sudoedit`, and restart
  `lerobot-container.service`. Updating local variables or user data does not
  change an existing VM's saved configuration.
- Do not delete EBS volumes until outputs are backed up. Retiring a Spot VM also
  requires reviewing its persistent Spot request; cancellation can terminate
  an associated stopped instance.

The regular workflow is: start VM, check startup, copy changed datasets, launch
or resume training, follow the log, then stop the VM.
