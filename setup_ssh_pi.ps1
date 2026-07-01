# One-time SSH setup: install your Windows public key on the Raspberry Pi.
#
# Prerequisites on the Pi:
#   - SSH enabled (Raspberry Pi Imager: "Enable SSH", or on the Pi:
#     sudo raspi-config -> Interface Options -> SSH -> Enable)
#   - Pi and this PC on the same network (or use the Pi's IP instead of .local)
#
# Usage:
#   .\setup_ssh_pi.ps1
#   .\setup_ssh_pi.ps1 -User pi -HostName 192.168.1.42
#   .\setup_ssh_pi.ps1 -User isump -HostName raspberrypi.local

param(
    [string]$User = $env:RPI_USER,
    [string]$HostName = $env:RPI_HOST,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_rsa.pub"
)

$ErrorActionPreference = "Stop"

if (-not $User) {
    $User = Read-Host "Pi login username (Bookworm: the user you created when imaging, not always 'pi')"
}
if (-not $HostName) {
    $HostName = Read-Host "Pi hostname or IP [raspberrypi.local]"
    if (-not $HostName) { $HostName = "raspberrypi.local" }
}
if (-not (Test-Path $KeyPath)) {
    Write-Host "No key at $KeyPath - generating one..."
    ssh-keygen -t rsa -b 4096 -f "$env:USERPROFILE\.ssh\id_rsa" -N '""'
    $KeyPath = "$env:USERPROFILE\.ssh\id_rsa.pub"
}

$target = "${User}@${HostName}"

Write-Host ""
Write-Host "Installing SSH key on $target (you will be asked for the Pi password once)..."
Write-Host ""

function Invoke-Ssh([string]$RemoteCmd) {
    & ssh $target $RemoteCmd
    if ($LASTEXITCODE -ne 0) {
        throw "ssh failed (exit $LASTEXITCODE)"
    }
}

function Show-SshHelp {
    Write-Host ""
    Write-Host "SSH failed. Check:" -ForegroundColor Yellow
    Write-Host "  - Pi is powered on and on the same network"
    Write-Host "  - SSH is enabled on the Pi"
    Write-Host "  - Hostname/IP is correct (try: ping $HostName)"
    Write-Host "  - Username is correct (Raspberry Pi OS Bookworm uses your imager username)"
}

try {
    Invoke-Ssh "mkdir -p ~/.ssh; chmod 700 ~/.ssh; touch ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys"

    & scp -q $KeyPath "${target}:.ssh/nsa_setup_key.pub"
    if ($LASTEXITCODE -ne 0) { throw "scp failed (exit $LASTEXITCODE)" }

    Invoke-Ssh "cat ~/.ssh/nsa_setup_key.pub >> ~/.ssh/authorized_keys; rm -f ~/.ssh/nsa_setup_key.pub; echo SSH key installed on; hostname"
}
catch {
    Show-SshHelp
    exit 1
}

Write-Host ""
Write-Host "Testing passwordless login..."
& ssh -o BatchMode=yes $target "hostname; uname -a"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Key login not working yet - re-run or check ~/.ssh on the Pi." -ForegroundColor Yellow
    exit 1
}

$sshConfig = "$env:USERPROFILE\.ssh\config"
$block = @"

# Raspberry Pi - added by RPi-Hardware-NSA setup_ssh_pi.ps1
Host rpi raspberrypi
    HostName $HostName
    User $User
    IdentityFile ~/.ssh/id_rsa
    ServerAliveInterval 30
    ServerAliveCountMax 4
"@

$existing = ""
if (Test-Path $sshConfig) {
    $existing = Get-Content $sshConfig -Raw
}
if ($existing -notmatch "(?m)^Host rpi\b") {
    Add-Content -Path $sshConfig -Value $block
    Write-Host "Wrote SSH config alias:  ssh rpi" -ForegroundColor Green
}
else {
    Write-Host "SSH config already has a 'Host rpi' entry - edit $sshConfig if HostName/User changed."
}

Write-Host ""
Write-Host "Done. Connect with:" -ForegroundColor Green
Write-Host "  ssh rpi"
