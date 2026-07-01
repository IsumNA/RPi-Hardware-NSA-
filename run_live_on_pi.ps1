# Sync outputs/model.pt to the Pi and open a terminal running live.py on the CSI camera.
#
# One-time setup (Windows):
#   .\setup_ssh_pi.ps1
#
# Then from the project root:
#   .\run_live_on_pi.ps1
#
# Or click LIVE TEST in the GUI (same flow automatically on Windows / AI server).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $Root
try {
    python -m nsa.pi_remote
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
