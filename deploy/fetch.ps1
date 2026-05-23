# Download results back to the laptop via scp.
#   pwsh deploy/fetch.ps1          # full: results/ + logs/ (includes models)
#   pwsh deploy/fetch.ps1 -Lite    # small: metrics/curve/eval/config/gif + logs
param([switch]$Lite)
. "$PSScriptRoot\_common.ps1"
$root = Split-Path $PSScriptRoot -Parent

if ($Lite) {
    Write-Host "Collecting lightweight artifacts on the server..."
    Invoke-Remote "cd $RemoteDir && bash deploy/remote_collect.sh > /dev/null"
    $localTar = Join-Path $root 'results_lite.tgz'
    Write-Host "Downloading results_lite.tgz ..."
    & scp @(Get-ScpArgs) "${Target}:$RemoteDir/results_lite.tgz" $localTar
    if ($LASTEXITCODE -ne 0) { throw "scp failed" }
    Invoke-Remote "rm -f $RemoteDir/results_lite.tgz"

    Push-Location $root
    try {
        tar -xzf $localTar
        if ($LASTEXITCODE -ne 0) { throw "local extract failed" }
        Remove-Item $localTar -Force
    }
    finally { Pop-Location }
    Write-Host "Done -> results/ and logs/ updated (lite)."
}
else {
    Write-Host "Downloading full results/ and logs/ (this may be large)..."
    & scp @(Get-ScpArgs) -r "${Target}:$RemoteDir/results" $root
    & scp @(Get-ScpArgs) -r "${Target}:$RemoteDir/logs" $root
    Write-Host "Done -> results/ and logs/ updated."
}
