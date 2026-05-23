# Upload the code/configs/docs to the server via scp (no git on the server).
# Packs a tarball (excluding results/venv/git), scp's it, extracts remotely,
# and strips CR line-endings from the shell scripts so they run on Linux.
#
#   pwsh deploy/upload.ps1
. "$PSScriptRoot\_common.ps1"
$root = Split-Path $PSScriptRoot -Parent
$tar = Join-Path $env:TEMP 'hw3_upload.tgz'

Push-Location $root
try {
    Write-Host "Packing repo -> $tar"
    tar --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' `
        --exclude='hw3' --exclude='.venv' --exclude='venv' `
        --exclude='deploy/server.env' --exclude='*.tgz' `
        -czf $tar `
        common pong vizdoom configs deploy `
        requirements.txt smoke_test.py `
        README.md EXPERIMENTS.md RESULTS.md REFERENCES.md plan.md .gitignore .gitattributes
    if ($LASTEXITCODE -ne 0) { throw "tar failed" }

    Write-Host "Creating remote dir: $RemoteDir"
    Invoke-Remote "mkdir -p $RemoteDir"

    Write-Host "Uploading to ${Target}:$RemoteDir ..."
    & scp @(Get-ScpArgs) $tar "${Target}:$RemoteDir/hw3_upload.tgz"
    if ($LASTEXITCODE -ne 0) { throw "scp failed" }

    Write-Host "Extracting on server (and normalizing shell scripts to LF)..."
    Invoke-Remote "cd $RemoteDir && tar -xzf hw3_upload.tgz && rm -f hw3_upload.tgz && sed -i 's/\r$//' deploy/*.sh && chmod +x deploy/*.sh"

    Write-Host "Upload complete -> ${Target}:$RemoteDir"
}
finally {
    if (Test-Path $tar) { Remove-Item $tar -Force }
    Pop-Location
}
