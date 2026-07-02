#!/usr/bin/env bash
# One-time SSH setup on the AI server → Raspberry Pi (for LIVE TEST).
#
# Usage:
#   ./setup_ssh_pi.sh
#   ./setup_ssh_pi.sh isump 10.3.195.124
#   RPI_USER=isump RPI_HOST=10.3.195.124 ./setup_ssh_pi.sh
#
# Then set config.yaml pi_live.ssh_host to "rpi" (alias) or user@ip directly.

set -euo pipefail

USER_NAME="${1:-${RPI_USER:-}}"
PI_HOST="${2:-${RPI_HOST:-}}"
KEY="${HOME}/.ssh/id_rsa"
PUB="${KEY}.pub"
SSH_CONFIG="${HOME}/.ssh/config"

if [[ -z "${USER_NAME}" ]]; then
  read -r -p "Pi login username: " USER_NAME
fi
if [[ -z "${PI_HOST}" ]]; then
  read -r -p "Pi IP or hostname [raspberrypi.local]: " PI_HOST
  PI_HOST="${PI_HOST:-raspberrypi.local}"
fi

mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"

if [[ ! -f "${PUB}" ]]; then
  echo "Generating SSH key at ${KEY}..."
  ssh-keygen -t rsa -b 4096 -f "${KEY}" -N ""
fi

TARGET="${USER_NAME}@${PI_HOST}"
echo ""
echo "Installing SSH key on ${TARGET} (enter Pi password once)..."
ssh-copy-id -i "${PUB}" "${TARGET}" || {
  echo "ssh-copy-id failed — trying manual install..."
  ssh "${TARGET}" "mkdir -p ~/.ssh && chmod 700 ~/.ssh"
  scp "${PUB}" "${TARGET}:.ssh/nsa_setup_key.pub"
  ssh "${TARGET}" "cat ~/.ssh/nsa_setup_key.pub >> ~/.ssh/authorized_keys && rm -f ~/.ssh/nsa_setup_key.pub && chmod 600 ~/.ssh/authorized_keys"
}

echo ""
echo "Testing passwordless login..."
ssh -o BatchMode=yes "${TARGET}" "hostname; uname -a"

BLOCK="
# Raspberry Pi — added by RPi-Hardware-NSA setup_ssh_pi.sh
Host rpi raspberrypi
    HostName ${PI_HOST}
    User ${USER_NAME}
    IdentityFile ~/.ssh/id_rsa
    ServerAliveInterval 30
    ServerAliveCountMax 4
"

if [[ -f "${SSH_CONFIG}" ]] && grep -qE '^Host rpi\b' "${SSH_CONFIG}"; then
  echo "SSH config already has Host rpi — edit ${SSH_CONFIG} if IP/user changed."
else
  printf '%s\n' "${BLOCK}" >> "${SSH_CONFIG}"
  chmod 600 "${SSH_CONFIG}"
  echo "Wrote SSH alias:  ssh rpi"
fi

echo ""
echo "Update config.yaml if needed:"
echo "  pi_live:"
echo "    ssh_host: rpi"
echo "    repo: ~/RPi-Hardware-NSA-"
echo ""
echo "Test live path:  python -m nsa.pi_remote --check"
echo "Done."
