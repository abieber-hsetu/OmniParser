#!/bin/bash
set -e 

echo "--- Initialize VM-System ---"

if [ -e /dev/kvm ]; then
    echo "KVM-Acceleration found."
else
    echo "Warning: KVM (Hardware-Virtualization) not found. VM will be very slow."
fi

CPU_MODEL="${CPU:-max}"

echo "CPU-Model set to: $CPU_MODEL"

mkdir -p /run/conf
mkdir -p /run/assets
mkdir -p /run/shm
touch /run/shm/status.html

echo "Initialization finished."