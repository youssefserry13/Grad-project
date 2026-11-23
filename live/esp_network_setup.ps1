# Nova: let the ESP32 (another device on the Wi-Fi) reach the bridge running inside WSL.
# Run in an ADMIN PowerShell after joining the lab Wi-Fi, and again after every reboot
# (the WSL IP changes) or Wi-Fi change (the laptop IP changes):
#
#   powershell -ExecutionPolicy Bypass -File C:\Grad\live\esp_network_setup.ps1
#   powershell -ExecutionPolicy Bypass -File C:\Grad\live\esp_network_setup.ps1 -Remove   # undo
#
# What it does:
#   1. portproxy  <Wi-Fi IP>:5001 -> <WSL IP>:5001   (the ESP talks to the laptop's Wi-Fi IP)
#      It listens on the Wi-Fi IP, not 0.0.0.0: WSL's own localhost relay (wslrelay) already
#      holds 127.0.0.1:5001 for Windows apps, and a 0.0.0.0 proxy on the same port can clash
#      with it. If you prefer 0.0.0.0, use:  -ListenAll
#   2. firewall: inbound TCP 5001 from the local subnet (the ESP), any network profile
#      (lab Wi-Fi is usually "Public").
#   3. firewall: inbound TCP 5002 from WSL only (172.16.0.0/12), so the bridge can fetch camera
#      frames from nova_live.py; the camera is not exposed to the Wi-Fi.
param([switch]$Remove, [switch]$ListenAll, [string]$WifiAlias = "Wi-Fi", [int]$Port = 5001)
$ErrorActionPreference = "Stop"

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this in an ADMIN PowerShell (right-click PowerShell > Run as administrator)."
}

# Remove every old proxy on this port (old Wi-Fi / WSL IPs) and our old firewall rules
netsh interface portproxy show v4tov4 | Select-String "^\s*(\S+)\s+$Port\s" | ForEach-Object {
    $addr = $_.Matches[0].Groups[1].Value
    netsh interface portproxy delete v4tov4 listenaddress=$addr listenport=$Port | Out-Null
}
Get-NetFirewallRule -DisplayName "Nova bridge $Port (ESP)", "Nova frame 5002 (WSL)" -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule
if ($Remove) { Write-Host "Removed Nova portproxy + firewall rules."; exit 0 }

$wslIp = ((wsl.exe hostname -I) -split "\s+" | Where-Object { $_ -match "^\d+\.\d+\.\d+\.\d+$" })[0]
$wifiIp = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias $WifiAlias -ErrorAction Stop |
           Where-Object { $_.IPAddress -notlike "169.254.*" } | Select-Object -First 1).IPAddress
if (-not $wslIp) { throw "Could not get the WSL IP (is WSL running?)" }
if (-not $wifiIp) { throw "No IPv4 address on adapter '$WifiAlias' (connected to Wi-Fi?)" }
$listen = if ($ListenAll) { "0.0.0.0" } else { $wifiIp }

netsh interface portproxy add v4tov4 listenaddress=$listen listenport=$Port connectaddress=$wslIp connectport=$Port | Out-Null
New-NetFirewallRule -DisplayName "Nova bridge $Port (ESP)" -Direction Inbound -Protocol TCP `
    -LocalPort $Port -RemoteAddress LocalSubnet -Action Allow -Profile Any | Out-Null
New-NetFirewallRule -DisplayName "Nova frame 5002 (WSL)" -Direction Inbound -Protocol TCP `
    -LocalPort 5002 -RemoteAddress 172.16.0.0/12 -Action Allow -Profile Any | Out-Null

netsh interface portproxy show v4tov4
Write-Host ""
Write-Host "WSL IP      : $wslIp"
Write-Host "Laptop Wi-Fi: $wifiIp"
Write-Host "ESP firmware: http://${wifiIp}:$Port/audio" -ForegroundColor Green
Write-Host "Test from a phone on the same Wi-Fi: http://${wifiIp}:$Port/last_answer"
