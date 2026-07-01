# Copy real PI_RAW from the AI machine to Desktop and point config.yaml at it.
#
# Usage (edit REMOTE if needed):
#   $env:PI_RAW_REMOTE = "you@ai-machine:/opt/datasets/PI_RAW"
#   .\fetch_pi_raw.ps1
#   .\fetch_pi_raw.ps1 -Remote "you@ai-machine:/opt/datasets/PI_RAW"
#   .\fetch_pi_raw.ps1 -Full   # entire dataset (large)
#
# If Desktop\PI_RAW already exists (from a prior fetch), NSA auto-uses it.

param(
    [string]$Remote = $env:PI_RAW_REMOTE,
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $Remote) {
    Write-Host "Set the AI machine SSH target, e.g.:"
    Write-Host '  $env:PI_RAW_REMOTE = "you@ai-machine:/opt/datasets/PI_RAW"'
    Write-Host "  .\fetch_pi_raw.ps1"
    Write-Host ""
    Write-Host "Or pass -Remote directly."
    exit 1
}

$args = @(
    "setup_denoise_hw_data.py",
    "--fetch", $Remote,
    "--desktop",
    "--write-config",
    "--no-clone"
)
if ($Full) { $args += "--full" }

Push-Location $Root
try {
    python @args
} finally {
    Pop-Location
}
